#include <aslam/cameras/distortion-doublesphere.h>

namespace aslam {
std::ostream& operator<<(std::ostream& out, const DoubleSphereDistortion& distortion) {
  distortion.printParameters(out, std::string(""));
  return out;
}

DoubleSphereDistortion::DoubleSphereDistortion(const Eigen::VectorXd& dist_coeffs)
: Base(dist_coeffs, Distortion::Type::kDoubleSphere) {
  CHECK(distortionParametersValid(dist_coeffs)) << dist_coeffs.transpose();
}

void DoubleSphereDistortion::distortUsingExternalCoefficients(
    const Eigen::VectorXd* dist_coeffs,
    Eigen::Vector2d* point,
    Eigen::Matrix2d* out_jacobian) const {
  CHECK_NOTNULL(point);

  double& x = (*point)(0);
  double& y = (*point)(1);

  // Use internal params if dist_coeffs==nullptr
  if(!dist_coeffs)
    dist_coeffs = &distortion_coefficients_;
  CHECK_EQ(dist_coeffs->size(), kNumOfParams) << "dist_coeffs: invalid size!";

  const double& k1 = (*dist_coeffs)(0); // The first double sphere distortion parameter (xi).
  const double& k2 = (*dist_coeffs)(1); // The second double sphere distortion parameter (alpha).

  double x2 = x*x;
  double y2 = y*y;
  double d1 = sqrt(x2 + y2 + 1.0);
  double d2 = sqrt(x2 + y2 + (k1 * d1 + 1.0));
  double s = 1.0 / (k2 * d2 + (1.0 - k2) * (k1 * d1 + 1.0));

  // Handle special case around image center.
  if (x2 + y2 < 1e-16) {
    // Keypoint remains unchanged.
    if(out_jacobian)
      out_jacobian->setZero();
    return;
  }

  if(out_jacobian) {
    double dd1_dx = x/d1;
    double dd1_dy = y/d1;
    double dd2_dx = (2 * x + 2 * dd1_dx * k1 * (d1 * k1 + 1)) / (2 * d2);
    double dd2_dy = (2 * y + 2 * dd1_dy * k1 * (d1 * k1 + 1)) / (2 * d2);

    double duf_du = s - x * s * s * (k2 * dd2_dx - k1 * dd1_dx * (k2 - 1));
    double duf_dv = -x * s * s * (k2 * dd2_dy - k1 * dd1_dy * (k2 - 1));
    double dvf_du = -y * s * s * (k2 * dd2_dx - k1 * dd1_dx * (k2 - 1));
    double dvf_dv = s - y * s * s * (k2 * dd2_dy - k1 * dd1_dy * (k2 - 1));

    *out_jacobian << duf_du, duf_dv,
                     dvf_du, dvf_dv;
  }

  x *= s;
  y *= s;
}

void DoubleSphereDistortion::distortParameterJacobian(
    const Eigen::VectorXd* dist_coeffs,
    const Eigen::Vector2d& point,
    Eigen::Matrix<double, 2, Eigen::Dynamic>* out_jacobian) const {
  CHECK_EQ(dist_coeffs->size(), kNumOfParams) << "dist_coeffs: invalid size!";
  CHECK_NOTNULL(out_jacobian);

  const double& k1 = (*dist_coeffs)(0); // The first double sphere distortion parameter (xi).
  const double& k2 = (*dist_coeffs)(1); // The second double sphere distortion parameter (alpha).

  const double& x = point(0);
  const double& y = point(1);

  double x2 = x*x;
  double y2 = y*y;
  double d1 = sqrt(x2 + y2 + 1.0);
  double d2 = sqrt(x2 + y2 + (k1 * d1 + 1.0));
  double s = 1.0 / (k2 * d2 + (1.0 - k2) * (k1 * d1 + 1.0));

  // Handle special case around image center.
  if (x2 + y2 < 1e-16) {
    out_jacobian->resize(2, kNumOfParams);
    out_jacobian->setZero();
    return;
  }

  double dd2_dk1 = (k1 * k1 + 2 * k1) / (2 * d2);

  const double duf_dk1 = -x * s * s * (k2 * dd2_dk1 - d1 * (k2 - 1));
  const double duf_dk2 = -x * s * s * (d2 - k1 * d1 - 1);

  const double dvf_dk1 = -y * s * s * (k2 * dd2_dk1 - d1 * (k2 - 1));
  const double dvf_dk2 = -y * s * s * (d2 - k1 * d1 - 1);

  out_jacobian->resize(2, kNumOfParams);
  *out_jacobian << duf_dk1, duf_dk2,
                   dvf_dk1, dvf_dk2;
}

void DoubleSphereDistortion::undistortUsingExternalCoefficients(const Eigen::VectorXd& dist_coeffs,
                                                               Eigen::Vector2d* point) const {
  CHECK_EQ(dist_coeffs.size(), kNumOfParams) << "dist_coeffs: invalid size!";
  CHECK_NOTNULL(point);

  const double& k1 = dist_coeffs(0); // The first double sphere distortion parameter (xi).
  const double& k2 = dist_coeffs(1); // The second double sphere distortion parameter (alpha).

  // Calculate distance from point to center.
  double r2 = point->squaredNorm();

  if (r2 == 0) {
    return;
  }

  double m_z = (1 - k2 * k2 * r2) / (k2 * sqrt(1 - (2 * k2 - 1) * r2) + 1 - k2);
  double s = (m_z * k1 + sqrt(m_z * m_z + (1 - k1 * k1) * r2)) / (m_z * m_z + r2);
  double r_u = s / (m_z * s - k1);

  (*point) *= r_u;
}

bool DoubleSphereDistortion::areParametersValid(const Eigen::VectorXd& parameters) {
  // Check the vector size.
  if (parameters.size() != kNumOfParams)
    return false;

  return true;
}

bool DoubleSphereDistortion::distortionParametersValid(const Eigen::VectorXd& dist_coeffs) const {
  return areParametersValid(dist_coeffs);
}

void DoubleSphereDistortion::printParameters(std::ostream& out, const std::string& text) const {
  const Eigen::VectorXd& distortion_coefficients = getParameters();
  CHECK_EQ(distortion_coefficients.size(), kNumOfParams) << "dist_coeffs: invalid size!";

  out << text << std::endl;
  out << "Distortion: (DoubleSphereDistortion) " << std::endl;
  out << "  k1 (xi): " << distortion_coefficients(0) << std::endl;
  out << "  k2 (alpha): " << distortion_coefficients(1) << std::endl;
}

} // namespace aslam
