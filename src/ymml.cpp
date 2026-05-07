#include "ymml.h"
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>

#define YMML_MEM_ALIGN 16

#define YMML_PAD(x, n) (((x) + (n) - 1) & ~((n) - 1))

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
                        : aligned_malloc(param.mem_size, YMML_MEM_ALIGN);
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

static struct ymml_tensor *ymml_new_tensor_impl(ymml_meta_data_arena *meta_arena,
                                    ymml_data_arena *data_arena,
                                    enum ymml_type type, int dims,
                                    uint64_t *ne){
  
}

struct ymml_tensor *ymml_new_tensor(ymml_meta_data_arena *meta_arena,
                                    ymml_data_arena *data_arena,
                                    enum ymml_type type, int dims,
                                    uint64_t *ne) {

  return ymml_new_tensor_impl(meta_arena,data_arena,type,dims,ne);
}
