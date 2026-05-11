#include "ymml.h"
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#define YMML_MEM_ALIGN 16
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
  ctx->mem_buffer = param.buffer
                        ? param.buffer
                        : aligned_malloc(mem_size, YMML_MEM_ALIGN);
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
                        : aligned_malloc(param.mem_size, YMML_MEM_ALIGN);
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

  memcpy(base_ptr, data, size);

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
