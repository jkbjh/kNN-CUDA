from libcpp cimport bool
import numpy as np
cimport numpy as np
import cython

assert sizeof(int) == sizeof(np.int32_t)

### CUBLAS
cdef extern from "src/knncuda.h":
    bool knn_cublas(void* refdata,
                    int ref_nb,
                    void* querydata,
                    int query_nb,
                    int dim,
                    int k,
                    void* out_distances,
                    void* out_indices,
    )


@cython.boundscheck(False)
@cython.wraparound(False)   
def cublas(np.ndarray[float, ndim=2, mode="c"] refdata not None,
           np.ndarray[float, ndim=2, mode="c"] querydata not None, int k):
    return f_cublas(np.asfortranarray(refdata), np.asfortranarray(querydata), k)	


@cython.boundscheck(False)
@cython.wraparound(False)   
def f_cublas(np.ndarray[float, ndim=2, mode="fortran"] refdata not None,
             np.ndarray[float, ndim=2, mode="fortran"] querydata not None, int k):
    # note that numpy defaults to C style row-major arrays, whereas CUDA uses fortran style column-major.
    
    cdef int ref_nb, dim, query_nb
    ref_nb, dim = refdata.shape[0], refdata.shape[1]
    query_nb = querydata.shape[0]

    cdef np.ndarray[np.float32_t, ndim=2] distances = np.zeros((query_nb, k), dtype=np.float32, order="F")
    cdef np.ndarray[np.int32_t, ndim=2] indices = np.zeros((query_nb, k), dtype=np.int32, order="F")

    result = knn_cublas(
    &refdata[0,0],
    ref_nb,
    &querydata[0,0],
    query_nb,
    dim,
    k,
    &distances[0,0],
    &indices[0,0],
    )

    assert result  # success calling knn_cublas?
    return indices, distances


### GLOBAL

cdef extern from "src/knncuda.h":
    bool knn_cuda_global(void* refdata,
                    int ref_nb,
                    void* querydata,
                    int query_nb,
                    int dim,
                    int k,
                    void* out_distances,
                    void* out_indices,
    )


@cython.boundscheck(False)
@cython.wraparound(False)   
def cuda_global(np.ndarray[float, ndim=2, mode="c"] refdata not None,
           np.ndarray[float, ndim=2, mode="c"] querydata not None, int k):
    return f_cuda_global(np.asfortranarray(refdata), np.asfortranarray(querydata), k)	


@cython.boundscheck(False)
@cython.wraparound(False)   
def f_cuda_global(np.ndarray[float, ndim=2, mode="fortran"] refdata not None,
             np.ndarray[float, ndim=2, mode="fortran"] querydata not None, int k):
    # note that numpy defaults to C style row-major arrays, whereas CUDA uses fortran style column-major.
    
    cdef int ref_nb, dim, query_nb
    ref_nb, dim = refdata.shape[0], refdata.shape[1]
    query_nb = querydata.shape[0]

    cdef np.ndarray[np.float32_t, ndim=2] distances = np.zeros((query_nb, k), dtype=np.float32, order="F")
    cdef np.ndarray[np.int32_t, ndim=2] indices = np.zeros((query_nb, k), dtype=np.int32, order="F")

    result = knn_cuda_global(
    &refdata[0,0],
    ref_nb,
    &querydata[0,0],
    query_nb,
    dim,
    k,
    &distances[0,0],
    &indices[0,0],
    )

    assert result  # success calling knn_cuda_global?
    return indices, distances


### TEXTURE

cdef extern from "src/knncuda.h":
    bool knn_cuda_texture(void* refdata,
                    int ref_nb,
                    void* querydata,
                    int query_nb,
                    int dim,
                    int k,
                    void* out_distances,
                    void* out_indices,
    )


@cython.boundscheck(False)
@cython.wraparound(False)   
def cuda_texture(np.ndarray[float, ndim=2, mode="c"] refdata not None,
           np.ndarray[float, ndim=2, mode="c"] querydata not None, int k):
    return f_cuda_texture(np.asfortranarray(refdata), np.asfortranarray(querydata), k)	


@cython.boundscheck(False)
@cython.wraparound(False)   
def f_cuda_texture(np.ndarray[float, ndim=2, mode="fortran"] refdata not None,
             np.ndarray[float, ndim=2, mode="fortran"] querydata not None, int k):
    # note that numpy defaults to C style row-major arrays, whereas CUDA uses fortran style column-major.
    
    cdef int ref_nb, dim, query_nb
    ref_nb, dim = refdata.shape[0], refdata.shape[1]
    query_nb = querydata.shape[0]

    cdef np.ndarray[np.float32_t, ndim=2] distances = np.zeros((query_nb, k), dtype=np.float32, order="F")
    cdef np.ndarray[np.int32_t, ndim=2] indices = np.zeros((query_nb, k), dtype=np.int32, order="F")

    result = knn_cuda_texture(
    &refdata[0,0],
    ref_nb,
    &querydata[0,0],
    query_nb,
    dim,
    k,
    &distances[0,0],
    &indices[0,0],
    )

    assert result  # success calling knn_cuda_texture?
    return indices, distances

