#pragma once

#include <BAT/BCModel.h>

#include <gsl/gsl_vector.h>
#include <gsl/gsl_multimin.h>

#include <string>

// ---------------------------------------------------------
class Tardis : public BCModel
{

public:
    /* using Vec = std::valarray<double>; */
    using Vec = std::vector<double>;

    // Constructor
    Tardis(const std::string& name, const std::string& fileName, unsigned run = 9, unsigned maxElements = 0);

    // Destructor
    ~Tardis();

    // Overload LogLikelihood to implement model
    virtual double LogLikelihood(const std::vector<double>& parameters)
    { return 0.0; }

    // Overload LogAprioriProbability if not using built-in 1D priors
    virtual double LogAPrioriProbability(const std::vector<double> & parameters)
    { return 0.0; }

    /**
     * Set state to update the prior such that minuit gets the correct
     * target density for prediction and compute the evidence.
     */
    void PreparePrediction();

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

    double PredictMedium(unsigned n, double X, double nu);

    double PredictVeryLarge(unsigned n, double X, double nu);

    void setScale(gsl_vector* v = nullptr);
    gsl_vector* initialValue() const;
    gsl_vector* stepSizes() const;
    gsl_multimin_fminimizer* minimizeSimplex(const double eps = 1e-3, const unsigned niter=150);
    gsl_multimin_fdfminimizer* minimizeLBFGS(gsl_vector* initial, const double eps = 0.5, const unsigned niter=100);

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

    double mean() const
    {
        const double sum = std::accumulate(samples.begin(), samples.end(), 0.0,
                [](double value, const Point& s1)
                {
                    return value + s1.en;
                });
        return sum / Nsamples();
    }

    void updateBlocks(gsl_matrix* m, std::vector<double>& powers,
            const double nu,
            const double alpha, const double beta, const double N);
    double logtarget(gsl_vector* v);
    gsl_matrix* hessian(gsl_vector* v);
    double logdet(gsl_matrix*);

    /// on log scale
    double Laplace(gsl_vector* v);

    enum class Target { Default, Gamma, NBGamma };

    Vec ReadData(const std::string& fileName, const std::string& dataSet, unsigned run, unsigned maxElements = 0);

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

    void FixPredicted(Target target, unsigned n, double X, double nu)
    {
        this->target = target;
        nPrediction = n;
        XPrediction = X;
        nuPrediction = nu;
    }

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
};
// ---------------------------------------------------------
std::ostream& operator<<(std::ostream&, gsl_vector*);
std::ostream& operator<<(std::ostream&, gsl_matrix*);
