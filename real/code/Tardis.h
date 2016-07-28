#pragma once

#include <Minuit2/MnUserParameters.h>
#include <Minuit2/FunctionMinimum.h>

#include <gsl/gsl_vector.h>
#include <gsl/gsl_multimin.h>

#include <string>
#include <vector>

class Tardis
{
public:
    /* using Vec = std::valarray<double>; */
    using Vec = std::vector<double>;

    // Constructor
    Tardis(const std::string& fileName, unsigned run = 9, unsigned maxElements = 0);

    // Destructor
    ~Tardis();

    /**
     * Set state to update the prior such that minuit gets the correct
     * target density for prediction and compute the evidence.
     */
    std::vector<double> PreparePrediction();

    /**
     * Predict X for small n at a given nu (=bin center).
     *
     * Algorithm: Start computing for N=n, then go to n-1,
     * n+1. Continue until contributions are negligible. Bear in mind
     * N>0! Stay on log scale as much as possible?
     *
     * @param Xmean If positive, use floor(X / Xmean) as starting guess for most likely N
     * @param precision Stop searching if contribution from current point is less than `precision * currentValue`
     */
    double PredictSmall(unsigned n, double X, double nu, double Xmean = -1, double precision = 1e-2);

    double PredictMedium(std::vector<double>& oldMode, unsigned n, double X, double nu);

    double PredictVeryLarge(unsigned n, double X, double nu);

    void setScale(const std::vector<double>& initial = {});
    std::vector<double> initialValue() const;
    gsl_vector* stepSizes(const unsigned ndim) const;

    class OptimOptions
    {
    public:
        static OptimOptions DefaultSimplex();
        static OptimOptions DefaultLBFGS();
        static OptimOptions DefaultMinuit();
        double eps, step_size, tol;
        unsigned iter_min, iter_max;
    private:
        OptimOptions(double eps,  double step_size, double tol,
                unsigned iter_min, unsigned iter_max);
    };

    gsl_multimin_fminimizer* minimizeSimplex(gsl_multimin_function, const std::vector<double>& initial, OptimOptions o);
    gsl_multimin_fminimizer* minimizeSimplex(const std::vector<double>& initial, OptimOptions = OptimOptions::DefaultSimplex());
//    gsl_multimin_fminimizer* minimizeSimplex(const std::vector<double>& initial = {});

    gsl_multimin_fdfminimizer* minimizeLBFGS(gsl_multimin_function_fdf, const std::vector<double>& initial, OptimOptions o);
    gsl_multimin_fdfminimizer* minimizeLBFGS(const std::vector<double>& initial = {}, OptimOptions = OptimOptions::DefaultLBFGS());
//    gsl_multimin_fdfminimizer* minimizeLBFGS(const std::vector<double>& initial = {});

    ROOT::Minuit2::FunctionMinimum minimizeMinuit(const std::vector<double>& initial = {},
            OptimOptions = OptimOptions::DefaultMinuit());

    void fitnb();

    /**
     * Compute sum of X in frequency bin
     */
    std::tuple<unsigned, double> SumX(double numin, double numax) const;

    /**
     * Polynomial as a function of nu with coefficients given in range
     */
    template <typename T>
    static double Polyn(T first, T last, const double& nu)
    {
        double res = 0.;
        double power = 1.;
        for (; first != last; ++first) {
            res += (*first) * power;
            power *= nu;
        }
        return res;
    }

    template <typename T>
    double alphaNu(const T& x, const double& nu)
    {
        using std::begin;
        return Polyn(begin(x), begin(x) + orderAlpha, nu);
    }

    template <typename T>
    double betaNu(const T& x, const double& nu)
    {
        using std::begin;
        return Polyn(begin(x) + orderAlpha, begin(x) + orderAlpha + orderBeta, nu);
    }

    double alphaNu_p(double* x, const double& nu)
    {
        return Polyn(x, x + orderAlpha, nu);
    }

    double betaNu_p(double* x, const double& nu)
    {
        return Polyn(x + orderAlpha, x + orderAlpha + orderBeta, nu);
    }

    size_t Nsamples() const
    {
        return samples.size();
    }

    double mean() const;
    void updateBlocks(gsl_matrix* m, std::vector<double>& powers,
            const double nu,
            const double alpha, const double beta, const double N);
    double logtarget(gsl_vector* v);
    gsl_matrix* hessian(const std::vector<double>& v);
    double logdet(gsl_matrix*);
    double log_likelihood(const std::vector<double>& v);

    /// on log scale
    double Laplace(const std::vector<double>& v);
    double Laplace(const std::vector<double>& v, const double logf);

    enum class Target { Default, Gamma, NBGamma, Undefined };

    static Vec ReadData(const std::string& fileName, const std::string& dataSet, unsigned run, unsigned maxElements = 0);

    /**
     * @param init use as initial position, write back updated results from minimization
     * @return for N posited events and n observed events, check if the contribution to res = sum_N P(X|N) is negligible with latest/ res < precision
     */
    bool SearchStep(unsigned N, unsigned n, double& res, double precision, Vec& init);

   /**
     * Minimum of polynomial given by coefficients in range. The argument is assumed to lie in [0,1].
     */
    static double MinPolyn(Vec::const_iterator begin, Vec::const_iterator end);

    struct Point
    {
         double en, nu;
    };

    void FixPredicted(Target target, unsigned n, double X, double nu);
    void set_target(Target target);

    void Unfix()
    {
        target = Target::Default;
        nPrediction = 0;
        XPrediction = -1;
        nuPrediction = -1;
    }

    unsigned orderN() const
    {
        return orderAlpha + orderBeta;
    }

    std::vector<Point> samples;

    unsigned orderAlpha, orderBeta;
    // unsigned orderEnmax;
    const unsigned npoints;
    const double nuMax, alphaMin, betaMin;
    double evidence;

    double scale;

    // parameter of prior on Poisson parameter
    double a;

    ///> prediction
    Target target;
    double nuPrediction, XPrediction;
    unsigned nPrediction, NPrediction;

    ///> optimization stats
    unsigned nCalls;

    /// parameters
    ROOT::Minuit2::MnUserParameters pars;
    unsigned GetNParameters() const
    {
        return pars.VariableParameters();
    }

};
// ---------------------------------------------------------
std::ostream& operator<<(std::ostream&, gsl_vector*);
std::ostream& operator<<(std::ostream&, gsl_matrix*);