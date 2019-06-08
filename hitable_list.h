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
// Portado para GPU por Luciano Soares <lpsoares@gmail.com>

#ifndef HITABLELISTH
#define HITABLELISTH

#include "hitable.h"

#include "sphere.h"
#include "material.h"


class hitable_list: public hitable  {
    public:
        hitable_list() {}
        hitable_list(hitable **l, int n);
        ~hitable_list();

        //__device__ __host__ virtual bool hit(const ray& r, float tmin, float tmax, hit_record& rec) const;
        __device__ __host__ bool hit(const ray& r, float tmin, float tmax, hit_record& rec) const;

        //hitable **list;
        sphere **list;

        int list_size;

        sphere **list_d;

};

hitable_list::hitable_list(hitable **l, int n) {

    list = (sphere **)l; 

    list_size = n; 

    cudaMalloc((void**)&list_d,n*sizeof(sphere *));
    sphere* p_hitlist_d[n];
    lambertian* p_lamb_d[n];
    for(int i = 0; i < n; ++i) {
        cudaMalloc(&p_lamb_d[i], list[i]->mat_ptr->size());
        cudaMemcpy(p_lamb_d[i],list[i]->mat_ptr,list[i]->mat_ptr->size(),cudaMemcpyHostToDevice);

        material *mat_ptr = list[i]->mat_ptr; // copia temporariamente para depois recuperar
        list[i]->mat_ptr=p_lamb_d[i];

        cudaMalloc(&p_hitlist_d[i], sizeof(sphere));
        cudaMemcpy(p_hitlist_d[i],list[i],sizeof(sphere),cudaMemcpyHostToDevice);

        list[i]->mat_ptr=mat_ptr;

    }
    cudaMemcpy(list_d, p_hitlist_d, sizeof(p_hitlist_d), cudaMemcpyHostToDevice);
    
}

hitable_list::~hitable_list() {
    for(int i = 0; i < list_size; ++i) {
        cudaFree(list[i]->mat_ptr);
        //cudaFree((sphere *)list_d[i]);
    }
    cudaFree(list_d);
}


__device__ __host__ bool hitable_list::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {

    hit_record temp_rec;
    bool hit_anything = false;
    double closest_so_far = t_max;

    for (int i = 0; i < list_size; i++) {
        if (list_d[i]->hit(r, t_min, closest_so_far, temp_rec)) {
            hit_anything = true;
            closest_so_far = temp_rec.t;
            rec = temp_rec;
        }
    }
    return hit_anything;
        
}

#endif

