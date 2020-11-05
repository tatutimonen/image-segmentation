#pragma once

#include <cstdlib>

extern "C" {
#include <immintrin.h>
#ifdef _WIN32
#include <malloc.h>
#endif
}

//----------------------------------------------------------------------------

namespace aligned_alloc
{
__m256d* alloc(size_t size);
void free(__m256d* ptr);
} // namespace aligned_alloc

//----------------------------------------------------------------------------
