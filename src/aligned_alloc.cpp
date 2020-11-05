#include "aligned_alloc.hpp"

using std::size_t;

namespace aligned_alloc
{
__m256d* alloc(size_t size)
{
#ifdef _WIN32
    return (__m256d*)_aligned_malloc(sizeof(__m256d)*size, sizeof(__m256d));
#else
    return (__m256d*)std::aligned_alloc(sizeof(__m256d), sizeof(__m256d)*size);
#endif
}

void free(__m256d* ptr)
{
#ifdef _WIN32
    _aligned_free(ptr);
#else
    return std::free(ptr);
#endif
}
} // namespace aligned_alloc