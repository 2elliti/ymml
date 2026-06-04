#pragma once

#include <cstddef>
#include <cstdint>
#ifdef YMML_SHARED
#define YMML_API __attribute__((visibility("default"))) extern
#else
#define YMML_API extern
#endif

#define YMML_MAX_DIMENSIONS 4
#define YMML_MAX_SRC 10
#define YMML_MAX_NAME 32
#define YMML_MAX_GRAPH_NODE 1024

typedef uint16_t ymml_fp16_t;

enum class ymml_type {
  YMML_NONE = 0,
  YMML_F16 = 1,
  YMML_F32 = 2,
  YMML_END = 3,
  YMML_INT64 = 4
};
enum class ymml_op { YMML_OP_ADD, YMML_OP_NONE };
enum class ymml_object_type {
  YMML_OBJ_TYP_TENSOR,
  YMML_OBJ_TYP_GRAPH,
  YMML_OBJ_TYP_OBJECT
};

// Stores offset, size and type of object in arena buffer.
struct ymml_object {
  size_t offset;
  size_t size;
  struct ymml_object *next;
  union {
    enum ymml_object_type meta_type;
    enum ymml_type data_type;
  } unified_type;
};

static const size_t YMML_OBJECT_SIZE = sizeof(struct ymml_object);

struct ymml_meta_data_param {
  size_t mem_size;
  void *buffer;
};

struct ymml_data_param {
  size_t mem_size;
  void *buffer;
};

struct ymml_meta_data_arena {
  size_t mem_size;
  void *mem_buffer;
  int total_objects = 0;

  // Keeps track of start and end of linked list
  struct ymml_object *begin;
  struct ymml_object *end;
};

struct ymml_data_arena {
  size_t mem_size;
  void *mem_buffer;
  int total_objects = 0;

  // Keeps track of start and end of linked list
  struct ymml_object *begin;
  struct ymml_object *end;
};

struct ymml_tensor {
  enum ymml_type type;
  char name[YMML_MAX_NAME];
  uint64_t ne[YMML_MAX_DIMENSIONS];
  uint32_t nb[YMML_MAX_DIMENSIONS];

  // Operation for this tensor.
  enum ymml_op op;

  // Parent tensors.
  struct ymml_tensor *src[YMML_MAX_SRC];

  // Stores the tensor data.
  void *data;

  // Question do we really need tensor data?
  // Or do we need a mechanism for storing info in other arena?
  struct ymml_object *data_object;

  uint8_t dims;

  bool visited;
};

static const size_t YMML_TENSOR_SIZE = sizeof(struct ymml_tensor);

YMML_API ymml_meta_data_arena *
ymml_init_meta_arena(ymml_meta_data_param &param);

YMML_API ymml_data_arena *ymml_init_data_arena(ymml_data_param &param);

YMML_API struct ymml_tensor *ymml_new_tensor(ymml_meta_data_arena *meta_arena,
                                             enum ymml_type type, int dims,
                                             uint64_t *ne);

YMML_API void ymml_fill_tensor_data(struct ymml_data_arena *data_arena,
                                    struct ymml_tensor *tensor, void *data,
                                    size_t size);

YMML_API struct ymml_object *ymml_get_tensor_data(struct ymml_tensor *tensor);

struct ymml_tensor *ymml_add(struct ymml_meta_data_arena *mdata,
                             struct ymml_data_arena *data,
                             struct ymml_tensor *a, struct ymml_tensor *b,
                             bool inplace);

/*
  Why ymml-graph?
  we  need  to convert  our tensor  and its resources into DAG (Directed acyclic
  graph) to execute  it.

  How are we going to do ?

  We  first decide what are the members that we need.

  - number  of nodes.
  - pointers to all the tensors.aa
*/

/*
How we tackling this?

*/
struct ymml_graph {
  // We need to  topo sort  all the tensors.
  // need a hash table for storing info wether we visited a node or not?
  size_t total_nodes;
  struct ymml_tensor *start_node;
  struct ymml_tensor **sorted_nodes;
  // Do i add a hash table in here?
};

static const size_t YMML_GRAPH_SIZE = sizeof(struct ymml_graph);
void ymml_build_forward_graph(struct ymml_graph *graph, struct ymml_tensor *tensor);
struct ymml_graph * ymml_new_graph(struct ymml_meta_data_arena *marena, struct ymml_data_arena *darena);
