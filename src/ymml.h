#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#ifdef YMML_SHARED
#define YMML_API __attribute__((visibility("default"))) extern
#else
#define YMML_API extern
#endif

#define YMML_MAX_DIMENSIONS 4
#define YMML_MAX_SRC 4
#define YMML_MAX_NAME 32
#define YMML_MAX_GRAPH_NODE 8192
#define YMML_MEM_ALIGN 64

#define YMML_ABORT(msg)                                                        \
  do {                                                                         \
    std::fprintf(stderr, "ymml abort: %s (%s:%d)\n", (msg), __FILE__,          \
                 __LINE__);                                                    \
    std::abort();                                                              \
  } while (0)

#define YMML_ASSERT(cond)                                                      \
  do {                                                                         \
    if (!(cond))                                                               \
      YMML_ABORT("assertion failed: " #cond);                                  \
  } while (0)

typedef uint16_t ymml_fp16_t;

enum class ymml_type {
  YMML_NONE = 0,
  YMML_F16 = 1,
  YMML_F32 = 2,
  YMML_INT64 = 3,
  YMML_END = 4
};

enum class ymml_op {
  YMML_OP_NONE = 0,
  YMML_OP_ADD,
};

enum class ymml_object_type {
  YMML_OBJ_TYP_TENSOR,
  YMML_OBJ_TYP_GRAPH,
  YMML_OBJ_TYP_OBJECT
};

struct ymml_object {
  size_t offset;
  size_t size;
  struct ymml_object *next;
  union {
    enum ymml_object_type meta_type;
    enum ymml_type data_type;
  } unified_type;
};

constexpr size_t YMML_OBJECT_SIZE = sizeof(struct ymml_object);

struct ymml_arena {
  size_t mem_size;
  void *mem_buffer;
  bool owns_buffer;
  int total_objects;
  struct ymml_object *begin;
  struct ymml_object *end;
};

struct ymml_arena_param {
  size_t mem_size;
  void *buffer;
};

using ymml_meta_data_arena = ymml_arena;
using ymml_data_arena = ymml_arena;
using ymml_meta_data_param = ymml_arena_param;
using ymml_data_param = ymml_arena_param;

struct ymml_tensor {
  enum ymml_type type;
  uint8_t dims;
  enum ymml_op op;
  uint64_t ne[YMML_MAX_DIMENSIONS];
  size_t nb[YMML_MAX_DIMENSIONS];
  void *data;
  struct ymml_tensor *src[YMML_MAX_SRC];
  struct ymml_object *data_object;
  char name[YMML_MAX_NAME];
};

constexpr size_t YMML_TENSOR_SIZE = sizeof(struct ymml_tensor);

struct ymml_graph {
  size_t total_nodes;
  size_t capacity;
  struct ymml_tensor *start_node;
  struct ymml_tensor **sorted_nodes;
  struct ymml_tensor **visited;
  size_t visited_cap;
  size_t visited_count;
};

constexpr size_t YMML_GRAPH_SIZE = sizeof(struct ymml_graph);

YMML_API ymml_arena *ymml_init_meta_arena(ymml_arena_param &param);
YMML_API ymml_arena *ymml_init_data_arena(ymml_arena_param &param);
YMML_API void ymml_free_arena(ymml_arena *arena);
YMML_API void ymml_arena_reset(ymml_arena *arena);

YMML_API struct ymml_tensor *ymml_new_tensor(ymml_arena *marena,
                                             enum ymml_type type, int dims,
                                             uint64_t *ne);

YMML_API void ymml_fill_tensor_data(ymml_arena *darena,
                                    struct ymml_tensor *tensor, void *data,
                                    size_t n_elements);

YMML_API struct ymml_object *ymml_get_tensor_data(struct ymml_tensor *tensor);

YMML_API bool ymml_is_contiguous(const struct ymml_tensor *t);
YMML_API size_t ymml_nbytes(const struct ymml_tensor *t);
YMML_API size_t ymml_nelements(const struct ymml_tensor *t);

YMML_API struct ymml_tensor *ymml_add(ymml_arena *marena, ymml_arena *darena,
                                      struct ymml_tensor *a,
                                      struct ymml_tensor *b, bool inplace);

YMML_API struct ymml_graph *ymml_new_graph(ymml_arena *marena);
YMML_API void ymml_build_forward_graph(struct ymml_graph *graph,
                                       struct ymml_tensor *root);
