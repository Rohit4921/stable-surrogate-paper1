/* O(n) linear solve for the Laplace approximation of a GP with Matern-5/2 kernel on sorted 1-D inputs.
 *
 * Given weights c_i (any sign) and vector b, returns
 *     f = (K^{-1} + diag(c))^{-1} b,
 *     log|I + K diag(c)| = sum_i log|1 + c_i s_i|  and its sign,
 * where K_ij = sf2 * Matern52(|x_i - x_j|; ell). K is represented by the state-space model of the
 * Matern-5/2 process (state = (f, f', f''), lam = sqrt(5)/ell). The solve is a Kalman filter with
 * scalar "information" updates followed by a Rauch-Tung-Striebel smoother for the means. Both are
 * block Gaussian elimination, so they are exact algebraic identities whenever no pivot 1 + c_i s_i
 * vanishes, also when some c_i < 0.
 * Returns 0 on success, 1 if a pivot vanished.
 */
#include <math.h>
#include <stdlib.h>
#include <string.h>

static void transition(double lam, double d, double A[9]) {
    /* A = exp(F d) = e^{-lam d} (I + N d + N^2 d^2 / 2), N = F + lam I nilpotent */
    double N[9] = {lam, 1, 0, 0, lam, 1, -lam * lam * lam, -3 * lam * lam, -2 * lam};
    double N2[9];
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) {
            double s = 0;
            for (int k = 0; k < 3; k++) s += N[3 * i + k] * N[3 * k + j];
            N2[3 * i + j] = s;
        }
    double e = exp(-lam * d);
    for (int i = 0; i < 9; i++) A[i] = e * (((i % 4) == 0 ? 1.0 : 0.0) + N[i] * d + 0.5 * N2[i] * d * d);
}

static void mat3_mul(const double *X, const double *Y, double *Z) {
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) {
            double s = 0;
            for (int k = 0; k < 3; k++) s += X[3 * i + k] * Y[3 * k + j];
            Z[3 * i + j] = s;
        }
}

static void mat3_mulT(const double *X, const double *Y, double *Z) { /* Z = X Y^T */
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) {
            double s = 0;
            for (int k = 0; k < 3; k++) s += X[3 * i + k] * Y[3 * j + k];
            Z[3 * i + j] = s;
        }
}

static int solve3(const double *M, const double *r, double *x) { /* x = M^{-1} r by cofactors */
    double a = M[0], b = M[1], c = M[2], d = M[3], e = M[4], f = M[5], g = M[6], h = M[7], k = M[8];
    double C00 = e * k - f * h, C01 = -(d * k - f * g), C02 = d * h - e * g;
    double det = a * C00 + b * C01 + c * C02;
    if (det == 0.0 || !isfinite(det)) return 1;
    double inv[9] = {C00, -(b * k - c * h), b * f - c * e,
                     C01, a * k - c * g, -(a * f - c * d),
                     C02, -(a * h - b * g), a * e - b * d};
    for (int i = 0; i < 3; i++) x[i] = (inv[3 * i] * r[0] + inv[3 * i + 1] * r[1] + inv[3 * i + 2] * r[2]) / det;
    return 0;
}

int ssm_solve(int n, const double *x, double sf2, double ell, const double *c, const double *b,
              double *fout, double *logdet, int *sign) {
    double lam = sqrt(5.0) / ell, l2 = lam * lam;
    double Pinf[9] = {sf2, 0, -sf2 * l2 / 3, 0, sf2 * l2 / 3, 0, -sf2 * l2 / 3, 0, sf2 * l2 * l2};
    double *mp = malloc(sizeof(double) * 3 * n), *mf = malloc(sizeof(double) * 3 * n);
    double *Pp = malloc(sizeof(double) * 9 * n), *Pf = malloc(sizeof(double) * 9 * n);
    double *At = malloc(sizeof(double) * 9 * n);
    int status = 0; double ld = 0; int sg = 1;
    for (int t = 0; t < n; t++) {
        double *m_ = mp + 3 * t, *P_ = Pp + 9 * t;
        if (t == 0) {
            m_[0] = m_[1] = m_[2] = 0; memcpy(P_, Pinf, sizeof Pinf);
        } else {
            double *A = At + 9 * t, D[9], T1[9], T2[9];
            transition(lam, x[t] - x[t - 1], A);
            const double *mprev = mf + 3 * (t - 1), *Pprev = Pf + 9 * (t - 1);
            for (int i = 0; i < 3; i++) m_[i] = A[3 * i] * mprev[0] + A[3 * i + 1] * mprev[1] + A[3 * i + 2] * mprev[2];
            for (int i = 0; i < 9; i++) D[i] = Pprev[i] - Pinf[i];        /* P- = Pinf + A (P+ - Pinf) A^T */
            mat3_mul(A, D, T1); mat3_mulT(T1, A, T2);
            for (int i = 0; i < 9; i++) P_[i] = Pinf[i] + T2[i];
            for (int i = 0; i < 3; i++)
                for (int j = i + 1; j < 3; j++) { double s = 0.5 * (P_[3 * i + j] + P_[3 * j + i]); P_[3 * i + j] = P_[3 * j + i] = s; }
        }
        double s = P_[0], den = 1.0 + c[t] * s;
        if (den == 0.0 || !isfinite(den)) { status = 1; break; }
        ld += log(fabs(den)); if (den < 0) sg = -sg;
        double kv[3] = {P_[0], P_[3], P_[6]}, inn = (b[t] - c[t] * m_[0]) / den;
        double *m = mf + 3 * t, *P = Pf + 9 * t;
        for (int i = 0; i < 3; i++) m[i] = m_[i] + kv[i] * inn;
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++) P[3 * i + j] = P_[3 * i + j] - c[t] * kv[i] * kv[j] / den;
    }
    if (status == 0) {
        double ms[3] = {mf[3 * (n - 1)], mf[3 * (n - 1) + 1], mf[3 * (n - 1) + 2]};
        fout[n - 1] = ms[0];
        for (int t = n - 2; t >= 0; t--) {
            /* m_s(t) = m_f(t) + P_f(t) A^T P_-(t+1)^{-1} (m_s(t+1) - m_-(t+1)) */
            double r[3], u[3], v[3];
            const double *A = At + 9 * (t + 1), *Pn = Pp + 9 * (t + 1), *mn = mp + 3 * (t + 1), *P = Pf + 9 * t, *m = mf + 3 * t;
            for (int i = 0; i < 3; i++) r[i] = ms[i] - mn[i];
            if (solve3(Pn, r, u)) { status = 1; break; }
            for (int i = 0; i < 3; i++) v[i] = A[i] * u[0] + A[3 + i] * u[1] + A[6 + i] * u[2];   /* A^T u */
            for (int i = 0; i < 3; i++) ms[i] = m[i] + P[3 * i] * v[0] + P[3 * i + 1] * v[1] + P[3 * i + 2] * v[2];
            fout[t] = ms[0];
        }
    }
    *logdet = ld; *sign = sg;
    free(mp); free(mf); free(Pp); free(Pf); free(At);
    return status;
}
