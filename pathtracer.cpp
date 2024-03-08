#include "pathtracer.h"

#include <iostream>

#include <Eigen/Dense>

#include <util/CS123Common.h>

#include <QtConcurrent>

using namespace Eigen;

bool USE_IMPORTANCE_SAMPLING = false;
bool USE_DOF = false;
bool USE_REFR_ATTENUATION = false;
bool USE_LOW_DISCREP_SAMP = false;
int RANDOM_SAMPLE_COUNTER = 0;

bool USE_STRATIFIED_SAMP = false;
int STRATIFIED_SIDE = 5;

PathTracer::PathTracer(int width, int height)
    : m_width(width), m_height(height)
{
}

void PathTracer::traceScene(QRgb *imageData, const Scene& scene)
{
    std::vector<Vector3f> intensityValues(m_width * m_height);
    Matrix4f invViewMat = (scene.getCamera().getScaleMatrix() * scene.getCamera().getViewMatrix()).inverse();


//    std::vector<std::tuple<int, int>> xyCoordinates;
//    for(int i = 0; i < m_height * m_width; ++i) {
//        xyCoordinates.push_back({i % m_width, i/m_height});
//    }
//    auto pixels = QtConcurrent::blockingMapped(xyCoordinates, [=](std::tuple<int, int> coord) {
//        int x = get<0>(coord);
//        int y = get<1>(coord);
//        Vector3f pixelVal = Eigen::Vector3f::Zero();
//        for(int i = 0; i < settings.samplesPerPixel; ++i) {
//            pixelVal += tracePixel(x, y, scene, invViewMat) / settings.samplesPerPixel;
//        }
//        return pixelVal;
//    });



    for(int y = 0; y < m_height; ++y) {
        #pragma omp parallel for
        for(int x = 0; x < m_width; ++x) {
            int offset = x + (y * m_width);
            if(USE_STRATIFIED_SAMP) {
                for (int i = 0; i < STRATIFIED_SIDE * STRATIFIED_SIDE; ++i) {
                    int inner_x = i % STRATIFIED_SIDE;
                    int inner_y = i / STRATIFIED_SIDE;
                    for(int j = 0; j < settings.samplesPerPixel / STRATIFIED_SIDE / STRATIFIED_SIDE; ++j) {
                        intensityValues[offset] += tracePixel(x + ((float) (inner_x) / (float) STRATIFIED_SIDE),
                                                              y + ((float) (inner_y) / (float) STRATIFIED_SIDE),
                                                              scene, invViewMat, 1.f / (float) STRATIFIED_SIDE) / settings.samplesPerPixel;
                    }
                }
            } else {
                for(int i = 0; i < settings.samplesPerPixel; ++i) {
                    intensityValues[offset] += tracePixel(x, y, scene, invViewMat) / settings.samplesPerPixel;
                }
            }

//            std::cout << "On pixel (" << x << ", " << y << ") out of " << m_width << "x" << m_height << std::endl;
//            intensityValues[offset] = tracePixel(x, y, scene, invViewMat);
        }
    }

    toneMap(imageData, intensityValues);
}

Vector3f PathTracer::tracePixel(float x, float y, const Scene& scene, const Matrix4f &invViewMatrix, float var)
{
    Vector3f p(0, 0, 0);
    float var_x, var_y;
    float var_r, var_theta;
    float lens_rad = 0.5f;
    float focal_depth = 3.f;
//    bool enable_dof = false;

    var_x = sampleNormalOrLowDisc() * var;
//    std::cout << var_x << ", " << var_y << std::endl;
    var_y = -(sampleNormalOrLowDisc()) * var;

    Vector3f d;

    if(!USE_DOF) {
        d = Vector3f((2.f * (x + var_x) / m_width) - 1, 1 - (2.f * (y + var_y) / m_height), - 1);
        d.normalize();
    } else {
        var_r = lens_rad * sqrt((sampleFloat()));
        var_theta = (sampleFloat()) * 2.f * M_PI;
        float aperature_x = var_r * cos(var_theta);
        float aperature_y = var_r * sin(var_theta);

        p = Eigen::Vector3f(aperature_x, aperature_y, 0);

        Vector3f d_initial((2.f * (x + var_x) / m_width) - 1, 1 - (2.f * (y + var_y) / m_height), -1.f);
        // Multiply the target point vector by the focal length
        // So that we can get the corresponding converging point on the plane
        Vector3f target_point = d_initial * focal_depth;

        // Then we get the direction from the scattered point to the target point
        // So that they will converge on the plane
        d = (target_point - p).normalized();
    }

    Ray r(p, d);
    r = r.transform(invViewMatrix);
    return traceRay(r, scene, true, 1.f, false);
}

std::pair<Vector3f, float> PathTracer::samplePointOnTriangle(Triangle* tri) {
    // https://chrischoy.github.io/research/barycentric-coordinate-for-mesh-sampling/#sampling-points-using-the-barycentric-coordinate
    // https://math.stackexchange.com/questions/3537762/random-point-in-a-triangle
    // https://math.stackexchange.com/questions/128991/how-to-calculate-the-area-of-a-3d-triangle
    float r1 = sampleFloat();
    float r2 = sampleFloat();
    float alpha = 1.f - sqrt(r1);
    float beta = (1.f - r2) * sqrt(r1);
    float gamma = r2 * sqrt(r1);
    Eigen::Vector3<Eigen::Vector3f> verts = tri->getVertices();
    float area = (verts.x() - verts.y()).cross(verts.z() - verts.y()).norm() / 2;
//    float area = (verts.x() - verts.y()).cross(verts.z() - verts.y()).norm();
    return std::pair {(alpha * verts.x()) + (beta * verts.y()) + (gamma * verts.z()), area};
}

//float PathTracer::getTriangleArea(Triangle* t) {
//    Eigen::Vector3<Eigen::Vector3f> verts = t->getVertices()
//}

Vector3f PathTracer::traceRay(const Ray& r, const Scene& scene, bool count_emitted, float current_ior, bool is_in_refractor)
{
    IntersectionInfo i;
    Ray ray(r);
    float threshold = settings.pathContinuationProb;

    if(scene.getIntersection(ray, &i)) {
        //** Example code for accessing materials provided by a .mtl file **
        const Triangle *t = static_cast<const Triangle *>(i.data); //Get the triangle in the mesh that was intersected
        const tinyobj::material_t& mat = t->getMaterial(); //Get the material of the triangle from the mesh
        const tinyobj::real_t *d = mat.diffuse; //Diffuse color as array of floats
        const std::string diffuseTex = mat.diffuse_texname; //Diffuse texture name
        Eigen::Vector3f intersectNormal = i.object->getNormal(i);
        Eigen::Vector3f matDif = Eigen::Vector3f(mat.diffuse);
        Eigen::Vector3f matSpec = Eigen::Vector3f(mat.specular);
        Eigen::Vector3f matEmiss = Eigen::Vector3f(mat.emission);
        Eigen::Vector3f L = Eigen::Vector3f::Zero();

        float r = sampleFloat();
        if (r < threshold) // If the random value tells us to continue
        {
            if(!settings.directLightingOnly) {
                if(mat.illum == 2){
                    bool isntSpecular = matSpec.isZero(0);
                    // Is intersect normal in object or world space.
                    std::pair<Eigen::Vector3f, float> w_i_pdf;
                    if(USE_IMPORTANCE_SAMPLING) {
                        if(isntSpecular) {
                            w_i_pdf = PathTracer::sampleNextDirDiff(intersectNormal, ray.d);
                        } else {
                            w_i_pdf = PathTracer::sampleNextDirSpec(intersectNormal, ray.d, mat.shininess);
                        }
                    } else {
                        w_i_pdf = PathTracer::sampleNextDir(intersectNormal); // Sample random dir
                    }
                    Eigen::Vector3f w_i = w_i_pdf.first; // Extract direction
                    float pdf = w_i_pdf.second; // Extract sample probabilities
                    const Ray r_i(i.hit, w_i); // Create ray in random direction
                    //        Eigen::Vector3f L_r = PathTracer::traceRay(r_i, scene) * t->getMaterial(). * intersectNormal.dot(d);
                    // TODO: mat.illum check
                    // Diffuse
                    Eigen::Vector3f L_r = Eigen::Vector3f::Zero();
                    Eigen::Vector3f recur = PathTracer::traceRay(r_i, scene, false, current_ior, false);

                    if(isntSpecular) { // If the material has no specular
                        L_r = recur * intersectNormal.dot(w_i)
                                / (threshold * pdf); // Divide by probability
                        if(matDif.x() <= 0.01 && matDif.y() <= 0.01 && matDif.z() <= 0.01) {
                            matDif = Vector3f(1, 1, 1);
                        }
                        L_r = L_r.cwiseProduct(matDif /  M_PI);
                    } else {
                        Eigen::Vector3f w_o = ray.d.normalized();
                        Eigen::Vector3f refl = w_i - 2 * w_i.dot(intersectNormal) * intersectNormal;
//                        Ray reflectedRay(i.hit, refl);
                        Eigen::Vector3f glossy_brdf = matSpec * std::pow(refl.dot(w_o), mat.shininess) * (mat.shininess + 2) / (2 * M_PI);
                        L_r = recur * std::clamp(intersectNormal.dot(w_i), 0.f, 1.f)
                                / (threshold * pdf);
                        L_r = L_r.cwiseProduct(glossy_brdf);
                    }

                    L += L_r;
                } else if (mat.illum == 5) {
                    Eigen::Vector3f w_o = ray.d.normalized();
                    Eigen::Vector3f w_i = w_o - 2 * w_o.dot(intersectNormal) * intersectNormal;
                    Ray reflectedRay(i.hit, w_i);
                    L += traceRay(reflectedRay, scene, true, current_ior, false) / threshold;
                } else if (mat.illum == 7) { // refractive
                    bool attenuateRefract = USE_REFR_ATTENUATION;
                    Eigen::Vector3f w_o = ray.d.normalized();
                    bool entering = (-w_o).dot(intersectNormal) > 0;
                    float n_i = entering ? 1 : mat.ior;
                    float n_t = entering ? mat.ior : 1;
                    Eigen::Vector3f incidenceNormal = entering ? intersectNormal : -intersectNormal;
                    float cos_theta_i = (-w_o).dot(incidenceNormal);
                    float cos_theta_t;
                    float determinant = 1.f - (pow((n_i / n_t), 2.f) * (1.f - pow(cos_theta_i, 2.f)));
                    float r0 = pow((n_i - n_t) / (n_i + n_t), 2.f); // variable required to calculate probability of reflection
                    float prob_to_refl = r0 + ((1 - r0) * pow((1 - cos_theta_i), 5.f));
                    float rand1 = sampleFloat();

                    if (rand1 > prob_to_refl && determinant >= 0) {
                        cos_theta_t = sqrt(determinant);
                        Eigen::Vector3f w_t = (n_i / n_t) * w_o + ((n_i / n_t) * cos_theta_i - cos_theta_t) * incidenceNormal;
                        Ray reflectedRay(i.hit, w_t);
//                        float attenuation = (!entering && attenuateRefract) ? pow((ray.o - i.hit).norm(), 2) + 1 : 1;
//                        float attenuation = (!entering && attenuateRefract) ? pow(((ray.o - i.hit).norm() + 1), 2) * 2 : 1;
                        float attenuation = (!entering && attenuateRefract) ? std::pow(M_E, (-(ray.o - i.hit).norm()) * mat.ior) : 1;
                        L += traceRay(reflectedRay, scene, true, n_t, !is_in_refractor) * attenuation / threshold;
                    } else {
                        Eigen::Vector3f w_i = w_o - 2 * w_o.dot(intersectNormal) * incidenceNormal;
                        Ray reflectedRay(i.hit, w_i);
                        L += traceRay(reflectedRay, scene, true, current_ior, false) / threshold;
                    }
                }
            }
        }

        // Direct Lighting

        std::vector<Triangle*> lights = scene.getEmissives();
        if(matEmiss.isZero(0)/* && mat.illum != 7*/)
        {
            for(int k = 0; k < settings.numDirectLightingSamples; ++k) {
                float r_triangle_i = rand() % lights.size(); // Not technically random
                for (int li = 0; li < lights.size(); ++li) {
                    Triangle* light = lights[li];
                    std::pair<Eigen::Vector3f, float> triSampleAndArea = samplePointOnTriangle(light);
                    Eigen::Vector3f pointOnLight = triSampleAndArea.first;
                    float lightArea = triSampleAndArea.second;
                    //                float lightSampleProb = 1 / lightArea;
                    Eigen::Vector3f dirToLight = pointOnLight - i.hit;
                    dirToLight = dirToLight.normalized();
                    const Ray r_l(i.hit, dirToLight);
                    IntersectionInfo potentialLightHit;
                    if(scene.getIntersection(r_l, &potentialLightHit)){
                        const Triangle* potentialLightTri = static_cast<const Triangle *>(potentialLightHit.data);
                        const tinyobj::material_t& potentialLightMat = potentialLightTri->getMaterial();//Get the material of the triangle from the mesh
                        Eigen::Vector3f lightNormal = potentialLightHit.object->getNormal(potentialLightHit);
                        float unobstructed = potentialLightTri == light ? 1 : 0;
                        float intersectNormCos = intersectNormal.dot(dirToLight);
                        float lightNormCos = std::clamp(lightNormal.dot(-dirToLight), 0.f, 1.f);
//                        float dist_squared = std::pow((i.hit - potentialLightHit.hit).norm(), 1.f);
                        float dist_squared = std::pow((i.hit - potentialLightHit.hit).norm(), 2.f);
                        Eigen::Vector3f L_d = Eigen::Vector3f(potentialLightMat.emission)
                                * intersectNormCos
                                * lightNormCos
                                * lightArea
                                //                            * lights.size()
                                / (float) lights.size()
                                //                            / lights.size()
                                / dist_squared
                                / (float) (settings.numDirectLightingSamples)
                                * unobstructed;

                        Eigen::Vector3f brdf = Eigen::Vector3f::Zero();

                        if(mat.illum == 2) {
                            if(matSpec.isZero(0)) {
                                brdf = matDif / M_PI;
                            } else {
                                Eigen::Vector3f w_o = ray.d.normalized();
                                Eigen::Vector3f refl = dirToLight - 2 * dirToLight.dot(intersectNormal) * intersectNormal;
                                brdf = matSpec * std::pow(refl.dot(w_o), mat.shininess) * (mat.shininess + 2) / (2 * M_PI);
                            }
                        }

                        L_d = L_d.cwiseProduct(brdf);
                        L += L_d;
                    }
                }
            }
        }

        // Emissive

        if(count_emitted) {
            L += matEmiss;
        }

        return L;
    } else {
        return Vector3f(0, 0, 0);
    }
}

std::pair<Vector3f, float> PathTracer::sampleNextDir(Eigen::Vector3f surfaceNormal) {
    float rad = 1;
    float xi_1, xi_2;
    xi_1 = sampleFloat();
    xi_2 = sampleFloat();
    float phi = 2 * M_PI * xi_1;
    float theta = acos(1 - xi_2);
    Eigen::Vector3f sampledDir = Eigen::Vector3f(
        rad * sin(theta) * cos(phi),
        rad * cos(theta),
        rad * sin(phi) * sin(theta)
    );
    Quaternion dir = Quaternionf::FromTwoVectors(Vector3f(0.f, 1.f, 0.f), surfaceNormal);
    return {(dir * sampledDir), 1 / (2 * M_PI)};
}

std::pair<Vector3f, float> PathTracer::sampleNextDirDiff(Eigen::Vector3f surfaceNormal, Eigen::Vector3f incomingRayDir) {
    float rad = 1;
    float xi_1, xi_2;
    xi_1 = sampleFloat();
    xi_2 = sampleFloat();
//    float pdf = std::clamp((float) ((-incomingRayDir).dot(surfaceNormal) / M_PI), 0.f, 1.f);
//    if(pdf < 0) {
//        std::cout << "asdfasdf" << std::endl;
//    }
    float phi = 2 * M_PI * xi_1;
    float theta = acos(sqrt(xi_2));
    Eigen::Vector3f sampledDir = Eigen::Vector3f(
        rad * sin(theta) * cos(phi),
        rad * cos(theta),
        rad * sin(phi) * sin(theta)
    );

    Quaternion dir = Quaternionf::FromTwoVectors(Vector3f(0.f, 1.f, 0.f), surfaceNormal);
    sampledDir = (dir * sampledDir);

    float pdf = ((sampledDir.normalized()).dot(surfaceNormal.normalized())) * (1 / M_PI);

    return {sampledDir, pdf};
}

std::pair<Vector3f, float> PathTracer::sampleNextDirSpec(Eigen::Vector3f surfaceNormal,
                                                         Eigen::Vector3f incomingRayDir,
                                                         float shininess) {
    float rad = 1;
    float xi_1, xi_2;
    xi_1 = sampleFloat();
    xi_2 = sampleFloat();
    float alpha = acos(pow(xi_1, (1.f / (shininess + 1.f))));
    float phi = 2 * M_PI * xi_2;
    Eigen::Vector3f sampledDir = Eigen::Vector3f(
        rad * sin(alpha) * cos(phi),
        rad * cos(alpha),
        rad * sin(phi) * sin(alpha)
    );
    Eigen::Vector3f refl = incomingRayDir - 2 * incomingRayDir.dot(surfaceNormal) * surfaceNormal; // TODO: Make helper for reflecting
    float pdf = ((shininess + 1) / (2 * M_PI)) * pow(cos(alpha), shininess);
    Quaternion dir = Quaternionf::FromTwoVectors(Vector3f(0.f, 1.f, 0.f), refl);
    return {(dir * sampledDir), pdf};
}

Vector3f PathTracer::getRadiance(Eigen::Vector3f point, Eigen::Vector3f dir) {

}

float PathTracer::sampleNormalOrLowDisc() {
    if (USE_LOW_DISCREP_SAMP) {
        float res = lowDiscrepancySample(RANDOM_SAMPLE_COUNTER);
        RANDOM_SAMPLE_COUNTER ++;
        if(RANDOM_SAMPLE_COUNTER >= settings.samplesPerPixel * 2) {
            RANDOM_SAMPLE_COUNTER = 0;
        }
        return res;
    } else {
        return sampleFloat();
    }
}

float PathTracer::sampleFloat() {
    return static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
}

float PathTracer::lowDiscrepancySample(int n, const int &base) {
//    float vanDerCorput(int n, const int &base = 2)
    float rand = 0, denom = 1, invBase = 1.f / base;
        while (n) {
            denom *= base;  //2, 4, 8, 16, etc, 2^1, 2^2, 2^3, 2^4 etc.
            rand += (n % base) / denom;
            n *= invBase;  //divide by 2
        }
        return rand;
}

float luminance(Vector3f v)
{
    return v.dot(Vector3f(0.2126f, 0.7152f, 0.0722f));
}

float luminance(float r, float g, float b) {
    return 0.2126f * r + 0.7152f * g + 0.0722f * b;
}

float reinhard_extended(float c, float max_white)
{
    float numerator = c * (1.0f + (c / (max_white * max_white)));
    return numerator / (1.0f + c);
}

void PathTracer::toneMap(QRgb *imageData, std::vector<Vector3f> &intensityValues) {
    float max_lum = 0;

    for(int y = 0; y < m_height; ++y) {
        for(int x = 0; x < m_width; ++x) {
            int offset = x + (y * m_width);
            float new_r, new_g, new_b;
            new_r = intensityValues[offset][0];
            new_g = intensityValues[offset][1];
            new_b = intensityValues[offset][2];
            float lum = luminance(new_r, new_g, new_b);
            max_lum = std::max(max_lum, lum);
        }
    }

    for(int y = 0; y < m_height; ++y) {
        for(int x = 0; x < m_width; ++x) {
            int offset = x + (y * m_width);
            float new_r, new_g, new_b;
            new_r = intensityValues[offset][0];
            new_g = intensityValues[offset][1];
            new_b = intensityValues[offset][2];
            float lum = luminance(new_r, new_g, new_b);
            new_r /= lum;
            new_g /= lum;
            new_b /= lum;
            float newLum = reinhard_extended(lum, max_lum);
            new_r *= newLum;
            new_g *= newLum;
            new_b *= newLum;
            new_r = std::clamp(new_r, 0.f, 1.f);
            new_g = std::clamp(new_g, 0.f, 1.f);
            new_b = std::clamp(new_b, 0.f, 1.f);
            new_r = std::pow(new_r, (1.f/2.2f));
            new_g = std::pow(new_g, (1.f/2.2f));
            new_b = std::pow(new_b, (1.f/2.2f));
            imageData[offset] = qRgb((int) (255 * new_r), (int) (255 * new_g), (int) (255 * new_b));
        }
    }

}
