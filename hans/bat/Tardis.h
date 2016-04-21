// ***************************************************************
// This file was created using the bat-project script.
// bat-project is part of Bayesian Analysis Toolkit (BAT).
// BAT can be downloaded from http://mpp.mpg.de/bat
// ***************************************************************

#ifndef __BAT__TARDIS__H
#define __BAT__TARDIS__H

#include <BAT/BCModel.h>

#include <string>
#include <valarray>

// This is a Tardis header file.
// Model source code is located in file Tardis/Tardis.cxx

// ---------------------------------------------------------
class Tardis : public BCModel
{

public:
    /* using Vec = std::valarray<double>; */
    using Vec = std::vector<double>;

    // Constructor
    Tardis(const std::string& name);

    // Destructor
    ~Tardis();

    // Overload LogLikelihood to implement model
    virtual double LogLikelihood(const std::vector<double>& parameters);

    // Overload LogAprioriProbability if not using built-in 1D priors
    virtual double LogAPrioriProbability(const std::vector<double> & parameters);

    virtual void CalculateObservables(const std::vector<double>& parameters);

    unsigned GetOrder() const
    { return order; }

    /**
     * Multiply the likelihood by this factor to avoid overflows.
     *
     * @param scale factor on the log(!) scale
     */
    void rescale(double scale)
    { this->scale = scale; }

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
     * @param precision Stop searching if contribution from current point is less than `precision * currentValue`
     */
    double PredictSmall(unsigned n, double X, double nu, double precision = 1e-2);

    /**
     * Compute sum of X in frequency bin
     */
    double SumX(double numin, double numax) const;

private:
    Vec ReadData(const std::string& fileName, const std::string& dataSet);

    /*
     * @return for N posited events and n observed events, check if the contribution to res = sum_N P(X|N) is negligible with latest/ res < precision
*/
    bool SearchStep(unsigned N, unsigned n, double& res, double precision);

    /**
     * Polynomial as a function of nu with coefficients given in range
     */
    double Polyn(Vec::const_iterator begin, Vec::const_iterator end, const double& nu);

    /**
     * Minimum of polynomial given by coefficients in range. The argument is assumed to lie in [0,1].
     */
    double MinPolyn(Vec::const_iterator begin, Vec::const_iterator end);

    struct Point
    {
         double en, nu;
    };

    std::vector<Point> samples;
    /* Vec energies, nus; */

    unsigned order;
    const unsigned npoints;
    const double nuMax, alphaMin, betaMin;
    double scale;
    double evidence;

    // parameter of prior on Poisson parameter
    double a;

    ///> prediction
    double nuPrediction, XPrediction;
    unsigned NPrediction;
};
// ---------------------------------------------------------

#endif
