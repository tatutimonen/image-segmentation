#include "AlignedAlloc.hpp"

namespace AlignedAlloc {
    
    __m256d* alloc(std::size_t size)
    {
#ifdef _WIN32
        return (__m256d*)_aligned_malloc(sizeof(__m256d)*size, sizeof(__m256d));
#else
        void* ptr = nullptr;
        if (posix_memalign(&ptr, sizeof(__m256d), sizeof(__m256d)*size)) {
            throw std::bad_alloc();
        }
        return (__m256d*)ptr;
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

} // namespace AlignedAlloc