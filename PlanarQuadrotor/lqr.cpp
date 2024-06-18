#include "lqr.h"


/* Discrete LQR */
Eigen::MatrixXf LQR(const Eigen::MatrixXf &A, const Eigen::MatrixXf &B, const Eigen::MatrixXf &Q, const Eigen::MatrixXf &R, double eps, u_int max_iter) {
    Eigen::MatrixXf A_T = A.transpose();
    Eigen::MatrixXf B_T = B.transpose();

    Eigen::MatrixXf P = Q;
    Eigen::MatrixXf P_old = P;
    Eigen::MatrixXf delta = Eigen::MatrixXf::Zero(P.rows(), P.cols());

    Eigen::MatrixXf K = Eigen::MatrixXf::Zero(A.rows(), A.cols());

    bool converged = false;

    for (u_int i = 0; i < max_iter; ++i) {

        P = A_T * P * A - A_T * P * B * (R + B_T * P * B).inverse() * B_T * P * A + Q;

        delta = P - P_old;
        if (fabs(delta.maxCoeff()) < eps) {
            converged = true;
            break;
        }
        P_old = P;
    }

    if (converged) {
        std::cout << "LQR: convergence reached.";
    } else {
        std::cout << "LQR: max iterations limit reached.";
    }

    K = (R + B_T * P * B).inverse() * (B_T * P * A);

    return K;
}

/*
Eigen::MatrixXf LQR(PlanarQuadrotor &quadrotor, float dt) {
    //LQR
    Eigen::MatrixXf Eye = Eigen::MatrixXf::Identity(6, 6);
    Eigen::MatrixXf A = Eigen::MatrixXf::Zero(6, 6);
    Eigen::MatrixXf A_discrete = Eigen::MatrixXf::Zero(6, 6);
    Eigen::MatrixXf B(6, 2);
    Eigen::MatrixXf B_discrete(6, 2);
    Eigen::MatrixXf Q = Eigen::MatrixXf::Identity(6, 6);
    Eigen::MatrixXf R = Eigen::MatrixXf::Identity(2, 2);
    Eigen::MatrixXf K = Eigen::MatrixXf::Zero(6, 6);
    Eigen::Vector2f input = quadrotor.GravityCompInput();

    Q.diagonal() << 0.004, 0.004, 400, 0.005, 0.045, 2 / 2 / M_PI;
    R.row(0) << 30, 7;
    R.row(1) << 7, 30;

    std::tie(A, B) = quadrotor.Linearize();
    A_discrete = Eye + dt * A;
    B_discrete = dt * B;

    return LQR(A_discrete, B_discrete, Q, R);
}
*/