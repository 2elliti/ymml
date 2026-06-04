#include "ymml.h"
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

#define YMML_MEM_ALIGN 16
#define YMML_MAX_GRAPH_TENSOR 1024
#define YMML_PLACEHOLDER_SIZE 16
#define YMML_PAD(x, n) (((x) + (n) - 1) & ~((n) - 1))

struct ymml_type_info_t {
  enum ymml_type type;
  size_t type_size;
};

static struct ymml_type_info_t type_info[static_cast<int>(
    ymml_type::YMML_END)] = {{ymml_type::YMML_F32, sizeof(float)},
                             {ymml_type::YMML_F16, sizeof(uint16_t)}};

static void *aligned_malloc(size_t mem_size, size_t alignment) {
  // lets make an extra space that we have to allocate.
  size_t total_allocate_space = mem_size + alignment + sizeof(void *);
  void *orignal_allocated_mem = malloc(total_allocate_space);

  // Now we have to find next alligned address.
  // for that we will use this formula: aligned = (raw + alignment - 1) &
  // ~(alignment - 1)

  // Keeping extra space for storing the original ptr.
  uintptr_t orignal_allocated_mem_addr =
      (uintptr_t)orignal_allocated_mem + sizeof(void *);

  uintptr_t aligned_mem =
      (orignal_allocated_mem_addr + alignment - 1) & ~(alignment - 1);
  ((void **)aligned_mem)[-1] = orignal_allocated_mem;
  return (void *)aligned_mem;
}

static void aligned_free(void *p) { free(((void **)p)[-1]); }

/*
 Aim: Has to return allocated arena for metadata.
*/
ymml_meta_data_arena *ymml_init_meta_arena(ymml_meta_data_param &param) {
  // Returns meta_data_arena,But  its possible that user might have already
  // allocated the  buffer. Why ? allocated arena might be on heterogeneous
  // hardware therefore  keep it  in checkkkkkkkkkkkk. Therefore  check if param
  // already  has allocated buffer or not.

  // First check if memsize is 0?
  if (param.mem_size == 0) {
    param.mem_size = YMML_MEM_ALIGN;
  }

  const size_t mem_size =
      param.buffer ? param.mem_size : YMML_PAD(param.mem_size, YMML_MEM_ALIGN);

  ymml_meta_data_arena *ctx =
      (ymml_meta_data_arena *)malloc(sizeof(ymml_meta_data_arena));
  ctx->mem_size = mem_size;
  ctx->mem_buffer =
      param.buffer ? param.buffer : aligned_malloc(mem_size, YMML_MEM_ALIGN);
  ctx->begin = nullptr;
  ctx->end = nullptr;
  ctx->total_objects = 0;
  assert(ctx->mem_buffer != nullptr);
  return ctx;
}

/*
  Aim: Returns allocated arena for payload.
*/

ymml_data_arena *ymml_init_data_arena(ymml_data_param &param) {
  // Returns the arena. Check if its already allocated or not.

  if (param.mem_size == 0) {
    param.mem_size = YMML_MEM_ALIGN;
  }
  const size_t mem_size =
      param.buffer ? param.mem_size : YMML_PAD(param.mem_size, YMML_MEM_ALIGN);

  ymml_data_arena *ctx = (ymml_data_arena *)malloc(sizeof(ymml_data_arena));
  ctx->mem_size = mem_size;
  ctx->mem_buffer = param.buffer
                        ? param.buffer
                        : aligned_malloc(mem_size, YMML_MEM_ALIGN);
  ctx->begin = nullptr;
  ctx->end = nullptr;
  ctx->total_objects = 0;
  return ctx;
}

static struct ymml_object *
ymml_new_meta_object(struct ymml_meta_data_arena *marena,
                     enum ymml_object_type type, size_t size) {
  struct ymml_object *object = marena->end;
  size_t last_obj_offset = object ? object->offset : 0;
  size_t last_obj_size = object ? object->size : 0;
  size_t end = last_obj_size + last_obj_offset;
  ymml_object *n_obj =
      (struct ymml_object *)((uintptr_t)marena->mem_buffer + end);
  marena->total_objects++;
  n_obj->offset = end + YMML_OBJECT_SIZE;
  n_obj->size = size;
  n_obj->unified_type.meta_type = type;
  n_obj->next = NULL;
  if (marena->end) {
    marena->end->next = n_obj;
  } else {
    marena->begin = n_obj;
  }
  marena->end = n_obj;
  return n_obj;
}

static struct ymml_tensor *
ymml_new_tensor_impl(ymml_meta_data_arena *meta_arena, enum ymml_type type,
                     int dims, uint64_t *ne) {
  // For this tensor allocate struct on meta data arena and buffer in data
  // arena.
  assert(dims >= 1 && dims <= YMML_MAX_DIMENSIONS);

  // start calculating total bytes needed to store in buffer.
  uint64_t total_elements = ne[0];
  for (int i = 1; i < dims; i++) {
    total_elements *= ne[i];
  }

  // Calculate number of bytes required for storing buffer data.
  uint64_t total_bytes =
      type_info[static_cast<uint64_t>(type)].type_size * total_elements;

  // We need a new way of thinking. WHat if i make or carve space out of the
  // buffer?
  struct ymml_object *tensor_object = ymml_new_meta_object(
      meta_arena, ymml_object_type::YMML_OBJ_TYP_TENSOR, YMML_TENSOR_SIZE);

  // Try getting tensor pointer.

  struct ymml_tensor *n_tensor =
      (struct ymml_tensor *)((uintptr_t)meta_arena->mem_buffer +
                             tensor_object->offset);
  for (uint32_t i = 0; i < dims; i++) {
    n_tensor->ne[i] = ne[i];
  }

  // Initialize unused dimensions as 1.
  for(uint32_t i = dims; i < YMML_MAX_DIMENSIONS; i++)n_tensor->ne[i] = 1;
  
  // Initialize all strides as 0.
  for(uint32_t i = 0; i < YMML_MAX_DIMENSIONS; i++)n_tensor->nb[i] = 0;

  // Initialize all src ptx as null.
  for(uint32_t i = 0; i < YMML_MAX_SRC; i++)n_tensor->src[i] = nullptr;
  n_tensor->dims = dims;
  n_tensor->type = ymml_type::YMML_NONE;
  n_tensor->visited = false;

  // Keep operations null here.
  n_tensor->op = ymml_op::YMML_OP_NONE;
  return n_tensor;
}

struct ymml_tensor *ymml_new_tensor(ymml_meta_data_arena *meta_arena,
                                    enum ymml_type type, int dims,
                                    uint64_t *ne) {

  return ymml_new_tensor_impl(meta_arena, type, dims, ne);
}

static struct ymml_object *
ymml_new_data_object(struct ymml_data_arena *data_arena, enum ymml_type type,
                     void *data, size_t size) {
  struct ymml_object *object = data_arena->end;
  size_t last_obj_size = object ? object->size : 0;
  size_t last_obj_offset = object ? object->offset : 0;
  size_t end = last_obj_offset + last_obj_size;

  struct ymml_object *n_obj =
      (struct ymml_object *)((uintptr_t)data_arena->mem_buffer + end);
  data_arena->total_objects++;
  n_obj->offset = end + YMML_OBJECT_SIZE;
  n_obj->size = size;
  n_obj->unified_type.data_type = type;
  char *base_ptr = (char *)data_arena->mem_buffer + n_obj->offset;

  // This here is worst thing, we ever did. Get this shit done.
  if(data != nullptr)memcpy(base_ptr, data, size);

  if (data_arena->end) {
    data_arena->end->next = n_obj;
  } else {
    data_arena->begin = n_obj;
  }

  data_arena->end = n_obj;
  return n_obj;
}

void ymml_fill_tensor_data(struct ymml_data_arena *data_arena,
                           struct ymml_tensor *tensor, void *data,
                           size_t size) {
  /*
    What is the aim for this function?
    What exactly do i have to do now?
    You have a data pointer.
    You have a total size.
  */

  // calculate the size of this shit.
  size_t total_bytes_req =
      type_info[static_cast<int>(tensor->type)].type_size * size;
  struct ymml_object *obj =
      ymml_new_data_object(data_arena, tensor->type, data, total_bytes_req);
  tensor->data_object = obj;
  tensor->data = (void *)(((char *)data_arena->mem_buffer + obj->offset));
}

struct ymml_object *ymml_get_tensor_data(struct ymml_tensor *tensor) {
  return tensor->data_object;
}


// returns a duplicate tensor.
static struct ymml_tensor *
ymml_tensor_duplicate(struct ymml_meta_data_arena *marena,
                      struct ymml_tensor *tensor) {
  return ymml_new_tensor(marena, tensor->type, tensor->dims, tensor->ne);
}

struct ymml_tensor *ymml_add(struct ymml_meta_data_arena *marena,
                             struct ymml_data_arena *data,
                             struct ymml_tensor *a, struct ymml_tensor *b,
                             bool inplace) {

  // we need to do some testing
  // need meta arena, whats the type? check the dims and ne.
  // We will give 2 options here, whether a user wants to go for inplace or not?
  
  struct ymml_tensor *ntensor = ymml_tensor_duplicate(marena, a);
  ntensor->op = ymml_op::YMML_OP_ADD;
  ntensor->src[0] = a;
  ntensor->src[1] = b;
  return ntensor;
}

static ymml_graph *ymml_new_graph_impl(struct ymml_meta_data_arena *marena){
  // Get the object (This keeps record of meta data).
  struct ymml_object *obj = ymml_new_meta_object(marena, ymml_object_type::YMML_OBJ_TYP_GRAPH, YMML_GRAPH_SIZE);
  // Build struct for graph and return.
  return (struct ymml_graph *)((uintptr_t)marena->mem_buffer + obj->offset);
}

struct ymml_graph * ymml_new_graph(struct ymml_meta_data_arena *marena, struct ymml_data_arena *darena){
  // Get the bare structure of graph.
  struct ymml_graph *ngraph = ymml_new_graph_impl(marena);

  // allocates space for sorted nodes.
  struct ymml_object *object = ymml_new_data_object(darena, ymml_type::YMML_INT64, nullptr, YMML_MAX_GRAPH_TENSOR);

  // Assign the allocated space to ngraph.
  ngraph->sorted_nodes = (struct ymml_tensor **)((uintptr_t)darena->mem_buffer + object->offset);
  
  // Keep start-node as nullptr.
  ngraph->start_node = nullptr;

  // Keep total nodes as 0
  ngraph->total_nodes = 0;

  return ngraph;
}

static void ymml_visit_parent_nodes(struct ymml_graph *graph, struct ymml_tensor *tensor){
  if(tensor == nullptr)return;

  for(uint8_t i = 0; i < YMML_MAX_SRC;i++){
    ymml_visit_parent_nodes(graph, tensor->src[i]);
  }
  graph->sorted_nodes[graph->total_nodes] = tensor;
  graph->total_nodes++;
}

void ymml_build_forward_graph(struct ymml_graph *graph, struct ymml_tensor *tensor){
  if(tensor == nullptr)return;
  ymml_visit_parent_nodes(graph, tensor);
}

