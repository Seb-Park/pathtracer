#ifndef PATHTRACER_H
#define PATHTRACER_H

#include <QImage>

#include "scene/scene.h"

struct Settings {
    int samplesPerPixel;
    bool directLightingOnly; // if true, ignore indirect lighting
    int numDirectLightingSamples; // number of shadow rays to trace from each intersection point
    float pathContinuationProb; // probability of spawning a new secondary ray == (1-pathTerminationProb)
};

class PathTracer
{
public:
    PathTracer(int width, int height);

    void traceScene(QRgb *imageData, const Scene &scene);
    Settings settings;

private:
    int m_width, m_height;

    void toneMap(QRgb *imageData, std::vector<Eigen::Vector3f> &intensityValues);

    Eigen::Vector3f tracePixel(float x, float y, const Scene &scene, const Eigen::Matrix4f &invViewMatrix, float var=1.f);
    Eigen::Vector3f traceRay(const Ray& r, const Scene &scene, bool count_emitted, float current_ior, bool is_in_refractor);
    Eigen::Vector3f getRadiance(Eigen::Vector3f point, Eigen::Vector3f dir);
    std::pair<Eigen::Vector3f, float> sampleNextDir(Eigen::Vector3f surfaceNormal);
    std::pair<Eigen::Vector3f, float> sampleNextDirDiff(Eigen::Vector3f surfaceNormal, Eigen::Vector3f incomingRayDir);
    std::pair<Eigen::Vector3f, float> sampleNextDirSpec(Eigen::Vector3f surfaceNormal, Eigen::Vector3f incomingRayDir, float shininess);
    std::pair<Eigen::Vector3f, float> samplePointOnTriangle(Triangle* tri);
    float sampleNormalOrLowDisc();
    float sampleFloat();
    float lowDiscrepancySample(int n, const int &base = 2);
    float getTriangleArea(Triangle* t);
};

#endif // PATHTRACER_H
