//==================================================================================================
// Written in 2016 by Peter Shirley <ptrshrl@gmail.com>
//
// To the extent possible under law, the author(s) have dedicated all copyright and related and
// neighboring rights to this software to the public domain worldwide. This software is distributed
// without any warranty.
//
// You should have received a copy (see file COPYING.txt) of the CC0 Public Domain Dedication along
// with this software. If not, see <http://creativecommons.org/publicdomain/zero/1.0/>.
//==================================================================================================

#ifndef RAYH
#define RAYH
#include "vec3.h"

class ray
{
    public:
        __device__ __host__ ray() {}
        __device__ __host__ ray(const vec3& a, const vec3& b) { A = a; B = b; }  
        __device__ __host__ vec3 origin() const       { return A; }
        __device__ __host__ vec3 direction() const    { return B; }
        __device__ __host__ vec3 point_at_parameter(float t) const { return A + t*B; }

        vec3 A;
        vec3 B;
};

#endif 


