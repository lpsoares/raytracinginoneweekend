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

#ifndef MATERIALH
#define MATERIALH 

struct hit_record;

#include "ray.h"
#include "hitable.h"

#include "curand_kernel.h"


__device__ __host__ float schlick(float cosine, float ref_idx) {
    float r0 = (((float)1)-ref_idx) / (((float)1)+ref_idx);
    r0 = r0*r0;
    return r0 + (((float)1)-r0)*pow((((float)1) - cosine),((float)5));
}

__device__ __host__ bool refract(const vec3& v, const vec3& n, float ni_over_nt, vec3& refracted) {
    vec3 uv = unit_vector(v);
    float dt = dot(uv, n);
    float discriminant = ((float)1.0) - ni_over_nt*ni_over_nt*(1-dt*dt);
    if (discriminant > ((float)0.0)) {
        refracted = ni_over_nt*(uv - n*dt) - n*sqrt(discriminant);
        return true;
    }
    else 
        return false;
}


__device__ __host__ vec3 reflect(const vec3& v, const vec3& n) {
     return v - ((float)2)*dot(v,n)*n;
}


__device__ vec3 random_in_unit_sphere() {
    vec3 p;

    curandState state;
    curand_init((unsigned long long)clock()+threadIdx.x+threadIdx.y, 0, 0, &state);
        
    do {
        p = ((float)2.0)*vec3(curand_uniform(&state),curand_uniform(&state),curand_uniform(&state)) - vec3(1,1,1);
    } while (p.squared_length() >= ((float)1.0));
    return p;  // Achei que ficou melhor (Luciano Soares)
}


class material  {
    public:
        //__device__  virtual bool scatter(const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered) const = 0;
        __device__ __host__ virtual int size() { return sizeof(material); }
        int type;
};

class lambertian : public material {
    public:
        lambertian(const vec3& a) : albedo(a) {type=0;}
        __device__  bool scatter(const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered) const  {
             vec3 target = rec.p + rec.normal + random_in_unit_sphere();
             scattered = ray(rec.p, target-rec.p);
             attenuation = albedo;
             return true;
        }
        __device__ __host__ virtual int size() { return sizeof(lambertian); }

        vec3 albedo;
};

class metal : public material {
    public:
        metal(const vec3& a, float f) : albedo(a) { if (f < 1) fuzz = f; else fuzz = 1;type=1; }
        __device__ bool scatter(const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered) const  {
            vec3 reflected = reflect(unit_vector(r_in.direction()), rec.normal);
            scattered = ray(rec.p, reflected + fuzz*random_in_unit_sphere());
            attenuation = albedo;
            return (dot(scattered.direction(), rec.normal) > ((float)0));
        }
        __device__ __host__ virtual int size() { return sizeof(metal); }
        vec3 albedo;
        float fuzz;
};

class dielectric : public material { 
    public:
        dielectric(float ri) : ref_idx(ri) {type=2;}
        __device__ bool scatter(const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered) const  {

            curandState state;
            curand_init((unsigned long long)clock()+threadIdx.x+threadIdx.y, 1, 0, &state);
        
             vec3 outward_normal;
             vec3 reflected = reflect(r_in.direction(), rec.normal);
             float ni_over_nt;
             attenuation = vec3(1.0, 1.0, 1.0); 
             vec3 refracted;
             float reflect_prob;
             float cosine;
             if (dot(r_in.direction(), rec.normal) > ((float)0.0)) {

                  outward_normal = -rec.normal;
                  
                  ni_over_nt = ref_idx;
         //         cosine = ref_idx * dot(r_in.direction(), rec.normal) / r_in.direction().length();
                  cosine = dot(r_in.direction(), rec.normal) / r_in.direction().length();
                  cosine = sqrt(1 - ref_idx*ref_idx*(1-cosine*cosine));
             }
             else {
                  outward_normal = rec.normal;
                  ni_over_nt = ((float)1.0) / ref_idx;
                  cosine = -dot(r_in.direction(), rec.normal) / r_in.direction().length();
             }
             if (refract(r_in.direction(), outward_normal, ni_over_nt, refracted)) 
                reflect_prob = schlick(cosine, ref_idx);
             else 
                reflect_prob = 1.0;
             if (curand_uniform(&state) < reflect_prob) 
                scattered = ray(rec.p, reflected);
             else 
                scattered = ray(rec.p, refracted);
             return true;
        }
        __device__ __host__ virtual int size() { return sizeof(dielectric); }

        float ref_idx;
};

#endif




