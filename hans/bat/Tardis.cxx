// ***************************************************************
// This file was created using the bat-project script.
// bat-project is part of Bayesian Analysis Toolkit (BAT).
// BAT can be downloaded from http://mpp.mpg.de/bat
// ***************************************************************

#include "Tardis.h"

#include <BAT/BCMath.h>

#include <TH1.h>
#include <TCanvas.h>

#include <H5Cpp.h>

#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>

using std::cout;
using std::endl;
using namespace H5;

namespace
{
    inline double logGamma(double x, double alpha, double beta)
    {
        return alpha * log(beta) - lgamma(alpha) + (alpha - 1) * log(x) - beta * x;
    }

    inline double logNegativeBinomial(const unsigned N, const unsigned n, const double a)
    {
        double tmp = N + n - a + 1;
        return lgamma(tmp) - lgamma(N + 1) - lgamma(n - a + 1) - tmp * std::log(2);
    }
}

// ---------------------------------------------------------
Tardis::Tardis(const std::string& name, const std::string& fileName, unsigned run)
    : BCModel(name),
      npoints(10),
      nuMax(0.5),
      alphaMin(1.0),
      betaMin(0.0),
      scale(0.0),
      evidence(0.0),
      a(0.5),
      nuPrediction(-1),
      XPrediction(-1),
      NPrediction(0)
{
    AddParameter("alpha0",  1, 2, "#alpha_{0}");
    AddParameter("alpha1", -3, 3, "#alpha_{1}");    // GetParameter(1).Fix(0);
#if 0
    AddParameter("alpha2", -10, 10, "#alpha_{2}");  // GetParameter(2).Fix(0);
#else
    AddParameter("alpha2", -30, 30, "#alpha_{2}");  // GetParameter(2).Fix(0);
    AddParameter("alpha3", -50, 50, "#alpha_{3}");  // GetParameter(3).Fix(0);
#endif
//    AddParameter("alpha4", -30, 30, "#alpha_{4}"); // GetParameter(3).Fix(0);

    order = GetNParameters();

    AddParameter("beta0",  0, 100, "#beta_{0}");
    AddParameter("beta1", -200, 50, "#beta_{1}");  // GetParameter(order + 1).Fix(0);
//    AddParameter("beta2",  0, 250, "#beta_{2}"); // GetParameter(order + 2).Fix(0);
    // AddParameter("beta3",  -300, 300, "#beta_{3}"); // GetParameter(order + 3).Fix(0);

    for (unsigned i = 0; i <= npoints; ++i) {
        double nu =  double(i) * nuMax / npoints;
        AddObservable(Form("alpha(%g)",nu), 1, 2,   Form("#alpha(%g)", nu));
    }
    for (unsigned i = 0; i <= npoints; ++i) {
        double nu =  double(i) * nuMax / npoints;
        AddObservable(Form("beta(%g)", nu), 0, 100, Form("#beta(%g)", nu));
    }

    AddObservable("Gamma(0.01|nu=0.2)", 25, 40);

    Vec energies = ReadData(fileName, "energies", run);
    Vec nus = ReadData(fileName, "nus", run);
    assert(energies.size() == nus.size());

    // remove negative energies
    samples.reserve(energies.size());
    for (unsigned i = 0; i < energies.size(); ++i) {
        if (energies[i] > 0) {
            samples.push_back(Point{energies[i], nus[i]});
        }
    }
    samples.shrink_to_fit();

    cout << "Parsed " << nus.size() << " elements from " << fileName
         << ", retain " << samples.size() << " positive-energy elements\n";

    // sort by frequency
    std::sort(samples.begin(), samples.end(),
              [](const Tardis::Point& s1, const Tardis::Point& s2)
              { return s1.nu < s2.nu; } );

    // rescale and flip the energies
    // rescale frequencies

    auto enOrder =
            [](const Tardis::Point& s1, const Tardis::Point& s2)
            { return s1.en < s2.en; };

    auto maxElem = std::max_element(samples.begin(), samples.end(), enOrder);

    const double maxEn = (1 + 1e-6) * maxElem->en;
    const double maxNu = samples.back().nu;
    cout << "Before data transformation" << endl;
    cout << "Max. energy = " << maxEn << endl;
    cout << "Max. frequency = " << maxNu << endl;

    for (auto& s : samples) {
        s.en = 1.0 - s.en / maxEn;
        s.nu /= maxNu;
    }

    // frequency sorting maintained but energy flipped => search again
    cout << "After data transformation" << endl;
    maxElem = std::max_element(samples.begin(), samples.end(), enOrder);
    cout << "Max. energy = " << maxElem->en << endl;
    cout << "Max. frequency = " << samples.back().nu << endl;

#if 0
    // plot data
    // create new histogram
    TH1D hist("data", ";x;N", 2000, 0.0, 1);
    // hist.SetStats(kFALSE);
    for (auto& s : samples)
        hist.Fill(s.nu);

    for (int i = 1; i <= hist.GetNbinsX(); ++i)
        cout << "bin " << i << ": [" << hist.GetBinLowEdge(i)
             << ", " << hist.GetBinLowEdge(i+1)
             << "], value = " << hist.GetBinContent(i)
             << endl;

    TCanvas c;
    hist.Draw();
    c.Print("samples.pdf");
#endif
}

// ---------------------------------------------------------
Tardis::~Tardis()
{
}

// ---------------------------------------------------------
double Tardis::LogLikelihood(const std::vector<double>& parameters)
{
#if 0
    cout << "like: (";
    std::copy(parameters.begin(), parameters.end(), std::ostream_iterator<double>(cout, ", " ));
    cout << ")" << endl;

    if (!std::isfinite(parameters.front())) {
        std::cout << "Got invalid parameter values in likelihood\n";
        std::copy(parameters.begin(), parameters.end(), std::ostream_iterator<double>(cout, " " ));
        cout << endl;
        throw 1;
    }
#endif
    double res = 0;

    auto alpha_start = parameters.begin();
    auto split = parameters.begin() + order;
    auto beta_end = parameters.end();

    const unsigned nsamples = samples.size();
    assert(nsamples <= samples.size());

#pragma omp parallel for reduction(+:res) schedule(static)
    for (unsigned i = 0; i < nsamples; ++i) {
        const auto& s = samples[i];

        // alpha(lambda_j)
        const double alphaj = Polyn(alpha_start, split, s.nu);
        const double betaj = Polyn(split, beta_end, s.nu);

        // use samples to check parameter space for a value inconsistent with prior
        if (alphaj <= alphaMin || betaj <= betaMin)
            res = -std::numeric_limits<double>::infinity();

        // cout << "alphaj " << alphaj << ", betaj "<< betaj << endl;

        const double extra = ::logGamma(s.en, alphaj, betaj);
        if (!std::isfinite(extra)) {
            cout << "res not finite at " << i << ", nu = " << s.nu << ", x = " << s.en << ", "<< alphaj << ", " << betaj << endl;
            std::copy(parameters.begin(), parameters.end(), std::ostream_iterator<double>(cout, " " ));
            cout << endl;
            throw 2;
        }
        // results are different from run to run with more than 2 threads
        // addition not commutative with floating point
        // change by 50% possible on linear scale if scale added only once
        // => apply part of scale each time to retain precision
        res += extra + this->scale / nsamples;
    }

    return res;
}

// ---------------------------------------------------------
double Tardis::LogAPrioriProbability(const std::vector<double> & parameters) {
#if 0
     // This returns the log of the prior probability for the parameters
     // If you use built-in 1D priors, don't uncomment this function.
     cout << "alpha_j = " << std::accumulate(parameters.begin(), parameters.begin() + order, 0) << ", beta_j = " << std::accumulate(parameters.begin() + order, parameters.end(), 0) << endl;

     std::copy(parameters.begin(), parameters.end(), std::ostream_iterator<double>(std::cout, " "));
     cout << endl;

     cout << std::accumulate(parameters.begin(), parameters.begin() + 1, 0.0) << endl;
     // TODO compute minimum of alpha(nu) for nu in [0,1] and test that it is > 1
     // alpha > 1
     if (std::accumulate(parameters.begin(), parameters.begin() + order, 0.0) <= 1.0)
         return -std::numeric_limits<double>::infinity();
     // beta > 0
     if (std::accumulate(parameters.begin() + order, parameters.end(), 0.0) <= 0.0)
         return -std::numeric_limits<double>::infinity();
#endif
    constexpr double invalid = -std::numeric_limits<double>::infinity();
    double valid = 0;

     // modify integrand for prediction
     if (nuPrediction > 0)
     {
         const double alpha = Polyn(parameters.begin(), parameters.begin() + order, nuPrediction);
         const double beta = Polyn(parameters.begin() + order, parameters.end(), nuPrediction);
         valid = logGamma(XPrediction, NPrediction * alpha, beta);
     }

     const double alphaMin = MinPolyn(parameters.begin(), parameters.begin() + order);
     if (alphaMin <= this->alphaMin)
         return invalid;

     const double betaMin = MinPolyn(parameters.begin() + order, parameters.end());
     if (betaMin <= this->betaMin)
         return invalid;

     // TODO
     // default: Jeffreys prior for Gamma shape parameter alpha and scale parameter beta
     // at some average frequency
#if 0
     const double nu = 0.2;
     double alpha = Polyn(parameters.begin(), parameters.begin() + order, nu);
     double beta = Polyn(parameters.begin() + order, parameters.end(), nu);
     return -log(beta) + 0.5 * log(alpha * PolyGamma(alpha) - 1.0);
#endif
     // uniform prior
     return valid;
}

void Tardis::CalculateObservables(const std::vector<double>& parameters)
{
    auto alpha_begin = parameters.begin();
    auto split = parameters.begin() + order;
    auto beta_end = parameters.end();

    // alpha and beta
    for (unsigned i = 0; i <= npoints; ++i) {
        double nu = double(i) * nuMax / npoints;
        GetObservable(i).Value(Polyn(alpha_begin, split, nu));
        GetObservable(i + npoints).Value(Polyn(split, beta_end, nu));
    }

    double en = 0.01;
    double nu = 0.2;
    double alpha = Polyn(alpha_begin, split, nu);
    double beta = Polyn(split, beta_end, nu);
    GetObservable(GetNObservables() - 1).Value(exp(alpha * log(beta) - lgamma(alpha) + (alpha - 1) * log(en) - beta * en));
}

double Tardis::Polyn(Vec::const_iterator first, Vec::const_iterator last, const double& nu)
{
    double res = 0.;
    double power = 1.;
    for (; first != last; ++first) {
        res += (*first) * power;
        power *= nu;
    }
    return res;
}

double Tardis::MinPolyn(Vec::const_iterator begin, Vec::const_iterator end)
{
    const int npar = std::distance(begin, end);
    double a0, a1, a2;
    switch (npar) {
    case 1: {
        return *begin;
    }
    case 2: {
        a0 = *begin;
        a1 = *(++begin);
        if (a1>= 0)
            return a0;
        else
            return a0 + a1;
    }
    case 3: {
        a0 = *begin;
        a1 = *(++begin);
        a2 = *(++begin);
        if ((a2 == 0 && a1 == 0) || (a2 == 0 && a1 > 0) || (a2 > 0 && a1 >= 0) || (a2 < 0 && a1 > -a2))
            return a0;
        if (a2 == 0 && a1 < 0)
            return a0 + a1;
        if ((a2 > 0 && a1 <= -2 * a2) || (a2 < 0 && a1 <= -a2))
            return a0 + a1 + a2;
        return (-a1 * a1 + 4 * a0 * a2)/ (4 * a2);
    }
    default:
        // TODO implement this
        // throw std::logic_error("Cubic and higher order not implemented");
        return 3;
    }
}

Tardis::Vec Tardis::ReadData(const std::string& fileName, const std::string& dataSet, unsigned run)
{
    // identify the run of tardis = row in column
//    static const hsize_t run = 9;

    H5File file(fileName.c_str(), H5F_ACC_RDONLY );
    DataSet dataset = file.openDataSet(dataSet);
    /*
     * Get dataspace of the dataset.
     */
    DataSpace dataspace = dataset.getSpace();

    /*
     * Get the dimension size of each dimension in the dataspace and
     * display them.
     */
    hsize_t dims_out[2];
    dataspace.getSimpleExtentDims( dims_out, NULL);
//    cout << "rank " << ndims << ", dimensions " <<
//        (unsigned long)(dims_out[0]) << " x " <<
//        (unsigned long)(dims_out[1]) << endl;

    // read one row from large matrix into 1D array
    static const int rankOut = 1;
    hsize_t N = dims_out[1];
    std::array<hsize_t, 2> offsetIn = {{ run, 0 }};
    std::array<hsize_t, 2> countIn = {{1, N}};
    dataspace.selectHyperslab(H5S_SELECT_SET, &countIn[0], &offsetIn[0]);

    // reserve memory to hold the data, resize so vector know how many
    // elements it has because hdf5 just writes directly to the heap buffer
    std::vector<double> buffer(N);
    // buffer.reserve(N);
    // buffer.resize(N);

    // define memory buffer layout
    DataSpace memspace(rankOut, &N);

    // where to write into the buffer: start at beginning and write up to the end
    hsize_t offset = 0;
    hsize_t count = N;
    memspace.selectHyperslab(H5S_SELECT_SET, &count, &offset);

    // read and possibly convert the data
    dataset.read(&buffer[0], PredType::NATIVE_DOUBLE, memspace, dataspace);

    return buffer;
}

void Tardis::PreparePrediction()
{
    Tardis::Vec v(GetNParameters(), 0.0);
    v[0] = 1.5;
    v[GetOrder()] = 60;
//    SetInitialPositions(v);

    FindMode(v);

    // change normalization to avoid overflow
    rescale(-LogEval(GetBestFitParameters()));
    // running minuit again might after redefining the
    // target density
    FindMode(GetBestFitParameters());
    evidence = Integrate(BCIntegrate::kIntLaplace);
}

bool Tardis::SearchStep(unsigned N, unsigned n, double& res, double precision, std::vector<double>& init)
{
    // N>0 required to search
    if (N == 0)
        return false;

    NPrediction = N;

    // run minuit manually assuming that current position and step
    // size are good gives a factor 2-3 in speed
    //    FindMode(GetBestFitParameters());
    auto& min = GetMinuit();
    min.SetVariableValues(&init[0]);
    min.Minimize();

    // copy result back
    std::copy(min.X(), min.X() + GetNParameters(), init.begin());

    const double latest = std::exp(logNegativeBinomial(N, n, a)) * Integrate(BCIntegrate::kIntLaplace);
    res += latest;

    cout << "total = " << res << ", P(" << XPrediction << "|" << N
    << ") = " << latest
    << endl;

    return (latest / res) > precision;
}

double Tardis::PredictSmall(unsigned n, double X, double nu, double Xmean, double precision)
{
    // fix frequency and X
    nuPrediction = nu;
    XPrediction = X;

    // start N at mode of NegativeBinomial if invalid Xmean given
    double guess = floor(n - a + 1);
    if (Xmean > 0)
        guess = X / Xmean;

    unsigned N = std::max(1.0, guess);
    NPrediction = N;

    double res = 0;

    // index difference down/up
    unsigned Nup = N, Ndown = N;

    std::vector<double> initUp(GetBestFitParameters());
    auto initDown = initUp;

    SearchStep(N, n, res, precision, initUp);

    // continue searching up or down
    bool goUp = true, goDown = true;

    // now move up or down
    while (goUp || goDown) {
        if (goUp)
            goUp = SearchStep(++Nup, n, res, precision, initUp);

        if (goDown)
            goDown = SearchStep(--Ndown, n,res, precision, initDown);
    }

    nuPrediction = -1;

    unsigned totalN = Nup - Ndown;
    if (Ndown == 0)
        --totalN;
    cout << "Total number of calls " << totalN << endl;

    return res / evidence;
}

std::tuple<unsigned, double> Tardis::SumX(double numin, double numax) const
{
    // find first element in bin
    auto min = std::lower_bound(samples.begin(), samples.end(), numin,
                                [](const Tardis::Point& s1, double value)
                                { return s1.nu < value; } );
    // first element outside of bin
    auto max = std::upper_bound(min, samples.end(), numax,
                                [](double value, const Tardis::Point& s1)
                                { return value < s1.nu; } );
    // sum energies in bin
    const double X = std::accumulate(min, max, 0.0,
        [](const double & res, const Tardis::Point& s2)
        { return res + s2.en; });

    unsigned n = std::distance(min, max);
    cout << "Found " << n
    << " elements in bin [" << numin << ", " << numax
    << "] with X = " << X << endl;

    return std::tuple<unsigned, double> {n, X};
}
