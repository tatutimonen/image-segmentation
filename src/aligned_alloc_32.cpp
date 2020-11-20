#include "aligned_alloc_32.hpp"

namespace aligned_alloc_32
{
__m256d* alloc(std::size_t size)
{
#ifdef _WIN32
    return (__m256d*)_aligned_malloc(sizeof(__m256d)*size, sizeof(__m256d));
#else
    void* p = nullptr;
    if (posix_memalign(&p, sizeof(__m256d), sizeof(__m256d)*size)) {
        throw std::bad_alloc();
    }
    return (__m256d*)p;
#endif
}

void free(__m256d* ptr)
{
#ifdef _WIN32
    _aligned_free(ptr);
#else
    std::free(ptr);
#endif
}
} // namespace aligned_alloc_32