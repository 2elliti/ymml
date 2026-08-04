#include "ymml.h"
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#define YMML_PAD(x, n) (((x) + (n) - 1) & ~((n) - 1))

struct ymml_type_info_t {
  enum ymml_type type;
  size_t type_size;
};

static const ymml_type_info_t type_info[static_cast<int>(ymml_type::YMML_END)] =
    {
        {ymml_type::YMML_NONE, 0},
        {ymml_type::YMML_F16, sizeof(uint16_t)},
        {ymml_type::YMML_F32, sizeof(float)},
        {ymml_type::YMML_INT64, sizeof(uint64_t)},
};

static void *aligned_malloc(size_t mem_size, size_t alignment) {
  size_t total = mem_size + alignment + sizeof(void *);
  void *raw = std::malloc(total);
  if (!raw)
    return nullptr;
  uintptr_t after_slot = (uintptr_t)raw + sizeof(void *);
  uintptr_t aligned =
      (after_slot + alignment - 1) & ~(uintptr_t)(alignment - 1);
  ((void **)aligned)[-1] = raw;
  return (void *)aligned;
}

static void aligned_free(void *p) {
  if (!p)
    return;
  std::free(((void **)p)[-1]);
}

static ymml_arena *arena_init_common(ymml_arena_param &param) {
  if (param.mem_size == 0)
    param.mem_size = YMML_MEM_ALIGN;

  const size_t mem_size =
      param.buffer ? param.mem_size : YMML_PAD(param.mem_size, YMML_MEM_ALIGN);

  auto *ctx = (ymml_arena *)std::malloc(sizeof(ymml_arena));
  YMML_ASSERT(ctx != nullptr);

  ctx->mem_size = mem_size;
  ctx->mem_buffer =
      param.buffer ? param.buffer : aligned_malloc(mem_size, YMML_MEM_ALIGN);
  ctx->owns_buffer = (param.buffer == nullptr);
  ctx->begin = nullptr;
  ctx->end = nullptr;
  ctx->total_objects = 0;

  YMML_ASSERT(ctx->mem_buffer != nullptr);
  return ctx;
}

ymml_arena *ymml_init_meta_arena(ymml_arena_param &param) {
  return arena_init_common(param);
}

ymml_arena *ymml_init_data_arena(ymml_arena_param &param) {
  return arena_init_common(param);
}

void ymml_free_arena(ymml_arena *arena) {
  if (!arena)
    return;
  if (arena->owns_buffer)
    aligned_free(arena->mem_buffer);
  std::free(arena);
}

void ymml_arena_reset(ymml_arena *arena) {
  arena->begin = nullptr;
  arena->end = nullptr;
  arena->total_objects = 0;
}

static ymml_object *arena_bump(ymml_arena *arena, size_t payload_size) {
  const size_t last_end =
      arena->end ? (arena->end->offset + arena->end->size) : 0;

  const size_t header_off = YMML_PAD(last_end, YMML_MEM_ALIGN);
  const size_t payload_off =
      YMML_PAD(header_off + YMML_OBJECT_SIZE, YMML_MEM_ALIGN);
  const size_t padded_size = YMML_PAD(payload_size, YMML_MEM_ALIGN);
  const size_t new_end = payload_off + padded_size;

  if (new_end > arena->mem_size) {
    std::fprintf(
        stderr, "ymml: arena OOM (need %zu, have %zu, requested payload=%zu)\n",
        new_end, arena->mem_size, payload_size);
    return nullptr;
  }

  auto *obj = (ymml_object *)((uintptr_t)arena->mem_buffer + header_off);
  obj->offset = payload_off;
  obj->size = padded_size;
  obj->next = nullptr;

  if (arena->end)
    arena->end->next = obj;
  else
    arena->begin = obj;
  arena->end = obj;
  arena->total_objects++;

  return obj;
}

static ymml_object *ymml_new_meta_object(ymml_arena *marena,
                                         ymml_object_type type, size_t size) {
  ymml_object *obj = arena_bump(marena, size);
  if (!obj)
    YMML_ABORT("meta arena OOM");
  obj->unified_type.meta_type = type;
  return obj;
}

static ymml_object *ymml_new_data_object(ymml_arena *darena, ymml_type type,
                                         const void *src, size_t size) {
  ymml_object *obj = arena_bump(darena, size);
  if (!obj)
    YMML_ABORT("data arena OOM");
  obj->unified_type.data_type = type;

  if (src != nullptr) {
    void *dst = (char *)darena->mem_buffer + obj->offset;
    std::memcpy(dst, src, size);
  }
  return obj;
}

size_t ymml_nelements(const ymml_tensor *t) {
  uint64_t n = 1;
  for (int i = 0; i < t->dims; i++)
    n *= t->ne[i];
  return (size_t)n;
}

size_t ymml_nbytes(const ymml_tensor *t) {
  return ymml_nelements(t) * type_info[(int)t->type].type_size;
}

bool ymml_is_contiguous(const ymml_tensor *t) {
  if (t->nb[0] != type_info[(int)t->type].type_size)
    return false;
  for (int i = 1; i < YMML_MAX_DIMENSIONS; i++) {
    if (t->nb[i] != t->nb[i - 1] * t->ne[i - 1])
      return false;
  }
  return true;
}

static ymml_tensor *ymml_new_tensor_impl(ymml_arena *marena, ymml_type type,
                                         int dims, uint64_t *ne) {
  YMML_ASSERT(dims >= 1 && dims <= YMML_MAX_DIMENSIONS);
  YMML_ASSERT((int)type > 0 && (int)type < (int)ymml_type::YMML_END);

  ymml_object *obj = ymml_new_meta_object(
      marena, ymml_object_type::YMML_OBJ_TYP_TENSOR, YMML_TENSOR_SIZE);

  auto *t = (ymml_tensor *)((uintptr_t)marena->mem_buffer + obj->offset);

  t->type = type;
  t->dims = (uint8_t)dims;
  t->op = ymml_op::YMML_OP_NONE;

  for (int i = 0; i < dims; i++)
    t->ne[i] = ne[i];
  for (int i = dims; i < YMML_MAX_DIMENSIONS; i++)
    t->ne[i] = 1;

  t->nb[0] = type_info[(int)type].type_size;
  for (int i = 1; i < YMML_MAX_DIMENSIONS; i++)
    t->nb[i] = t->nb[i - 1] * t->ne[i - 1];

  for (int i = 0; i < YMML_MAX_SRC; i++)
    t->src[i] = nullptr;

  t->data = nullptr;
  t->data_object = nullptr;
  t->name[0] = '\0';

  return t;
}

ymml_tensor *ymml_new_tensor(ymml_arena *marena, ymml_type type, int dims,
                             uint64_t *ne) {
  return ymml_new_tensor_impl(marena, type, dims, ne);
}

void ymml_fill_tensor_data(ymml_arena *darena, ymml_tensor *tensor, void *data,
                           size_t n_elements) {
  YMML_ASSERT(n_elements == ymml_nelements(tensor));
  const size_t nbytes = type_info[(int)tensor->type].type_size * n_elements;

  ymml_object *obj = ymml_new_data_object(darena, tensor->type, data, nbytes);
  tensor->data_object = obj;
  tensor->data = (char *)darena->mem_buffer + obj->offset;
}

ymml_object *ymml_get_tensor_data(ymml_tensor *tensor) {
  return tensor->data_object;
}

static ymml_tensor *ymml_tensor_dup(ymml_arena *marena, ymml_tensor *t) {
  return ymml_new_tensor(marena, t->type, t->dims, t->ne);
}

ymml_tensor *ymml_add(ymml_arena *marena, ymml_arena * /*darena*/,
                      ymml_tensor *a, ymml_tensor *b, bool inplace) {
  YMML_ASSERT(a && b);
  YMML_ASSERT(a->type == b->type);
  for (int i = 0; i < YMML_MAX_DIMENSIONS; i++)
    YMML_ASSERT(a->ne[i] == b->ne[i]);

  ymml_tensor *out;
  if (inplace) {
    out = ymml_tensor_dup(marena, a);
    out->data = a->data;
    out->data_object = a->data_object;
  } else {
    out = ymml_tensor_dup(marena, a);
  }
  out->op = ymml_op::YMML_OP_ADD;
  out->src[0] = a;
  out->src[1] = b;
  return out;
}

static inline size_t hash_ptr(const void *p) {
  uintptr_t x = (uintptr_t)p;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  x = x ^ (x >> 31);
  return (size_t)x;
}

static bool visited_insert(ymml_graph *g, ymml_tensor *t) {
  const size_t mask = g->visited_cap - 1;
  size_t i = hash_ptr(t) & mask;
  for (;;) {
    if (g->visited[i] == nullptr) {
      g->visited[i] = t;
      g->visited_count++;
      return true;
    }
    if (g->visited[i] == t)
      return false;
    i = (i + 1) & mask;
  }
}

static void ymml_visit(ymml_graph *g, ymml_tensor *t) {
  if (!t)
    return;
  if (!visited_insert(g, t))
    return;

  for (int i = 0; i < YMML_MAX_SRC; i++)
    ymml_visit(g, t->src[i]);

  YMML_ASSERT(g->total_nodes < g->capacity);
  g->sorted_nodes[g->total_nodes++] = t;
}

ymml_graph *ymml_new_graph(ymml_arena *marena) {
  ymml_object *gobj = ymml_new_meta_object(
      marena, ymml_object_type::YMML_OBJ_TYP_GRAPH, YMML_GRAPH_SIZE);
  auto *g = (ymml_graph *)((uintptr_t)marena->mem_buffer + gobj->offset);

  const size_t cap = YMML_MAX_GRAPH_NODE;
  ymml_object *nodes_obj =
      ymml_new_meta_object(marena, ymml_object_type::YMML_OBJ_TYP_OBJECT,
                           cap * sizeof(ymml_tensor *));
  g->sorted_nodes =
      (ymml_tensor **)((uintptr_t)marena->mem_buffer + nodes_obj->offset);
  g->capacity = cap;
  g->total_nodes = 0;
  g->start_node = nullptr;

  size_t vcap = 1;
  while (vcap < cap * 2)
    vcap <<= 1;
  ymml_object *vobj =
      ymml_new_meta_object(marena, ymml_object_type::YMML_OBJ_TYP_OBJECT,
                           vcap * sizeof(ymml_tensor *));
  g->visited = (ymml_tensor **)((uintptr_t)marena->mem_buffer + vobj->offset);
  g->visited_cap = vcap;
  g->visited_count = 0;
  std::memset(g->visited, 0, vcap * sizeof(ymml_tensor *));

  return g;
}

void ymml_build_forward_graph(ymml_graph *graph, ymml_tensor *root) {
  if (!root)
    return;

  graph->total_nodes = 0;
  graph->visited_count = 0;
  std::memset(graph->visited, 0, graph->visited_cap * sizeof(ymml_tensor *));

  graph->start_node = root;
  ymml_visit(graph, root);
}
