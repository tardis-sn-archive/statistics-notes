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
#include <valarray>

using std::cout;
using std::endl;
using namespace H5;

// ---------------------------------------------------------
Tardis::Tardis(const std::string& name)
    : BCModel(name),
      order(4),
      npoints(10),
      nuMax(0.5),
      alphaMin(1.0),
      betaMin(0.0)
{
    AddParameter("alpha0",  1, 2, "#alpha_{0}");
    AddParameter("alpha1", -3, 3, "#alpha_{1}"); // GetParameter(1).Fix(0);
    AddParameter("alpha2", -20, 20, "#alpha_{2}"); // GetParameter(2).Fix(0);
    AddParameter("alpha3", -20, 20, "#alpha_{3}"); // GetParameter(3).Fix(0);
    // AddParameter("alpha4",  0, 20, "#alpha_{4}"); // GetParameter(3).Fix(0);

    AddParameter("beta0",  0, 100, "#beta_{0}");
    AddParameter("beta1", -200, 50, "#beta_{1}"); // GetParameter(order + 1).Fix(0);
    AddParameter("beta2",  0, 200, "#beta_{2}"); // GetParameter(order + 2).Fix(0);
    // AddParameter("beta3",  -300, 300, "#beta_{3}"); // GetParameter(order + 3).Fix(0);

    for (unsigned i = 0; i <= npoints; ++i) {
        double nu =  double(i) * nuMax / npoints;
        AddObservable(Form("alpha(%g)",nu), 1, 2,   Form("#alpha(%g)", nu));
    }
    for (unsigned i = 0; i <= npoints; ++i) {
        double nu =  double(i) * nuMax / npoints;
        AddObservable(Form("beta(%g)", nu), 0, 100, Form("#beta(%g)", nu));
    }

    static const char* fileName = "../../posterior/real_tardis_250.h5";
    Vec energies = ReadData(fileName, "energies");
    Vec nus = ReadData(fileName, "nus");
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

    // rescale and flip the energies
    // rescale frequencies
    // auto maxElem = std::max_element(energies.begin(), energies.end());
    auto maxElem = std::max_element(samples.begin(), samples.end(),
                                    [](const Tardis::Point& s1, const Tardis::Point& s2)
                                    { return s1.en < s2.en; } );
    auto maxElemNu = std::max_element(samples.begin(), samples.end(),
                                    [](const Tardis::Point& s1, const Tardis::Point& s2)
                                    { return s1.nu < s2.nu; } );
    const double maxEn = (1 + 1e-6) * maxElem->en;
    const double maxNu = maxElemNu->nu;
    cout << "Max. energy = " << maxEn << endl;
    cout << "Max. frequency = " << maxNu << endl;
    for (auto& s : samples) {
        s.en = 1.0 - s.en / maxEn;
        s.nu /= maxNu;
    }
    maxElem = std::max_element(samples.begin(), samples.end(),
                                    [](const Tardis::Point& s1, const Tardis::Point& s2)
                                    { return s1.en < s2.en; } );
    maxElemNu = std::max_element(samples.begin(), samples.end(),
                                    [](const Tardis::Point& s1, const Tardis::Point& s2)
                                    { return s1.nu < s2.nu; } );
    cout << "Max. energy = " << maxElem->en << endl;
    cout << "Max. frequency = " << maxElemNu->nu << endl;


    // plot data
    // create new histogram
    TH1D hist("data", ";x;N", 100, 0.0, 1);
    // hist.SetStats(kFALSE);
    for (auto& s : samples)
        hist.Fill(s.nu);

    TCanvas c;
    hist.Draw("");
    c.Print("samples.pdf");
}

// ---------------------------------------------------------
Tardis::~Tardis()
{
}

Tardis::Vec Tardis::ReadData(const std::string& fileName, const std::string& dataSet)
{
    // identify the run of tardis = row in column
    static const hsize_t run = 9;

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
    std::array<hsize_t, 2> offsetIn = {run, 0};
    std::array<hsize_t, 2> countIn = {1, N};
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

// ---------------------------------------------------------
double Tardis::LogLikelihood(const std::vector<double>& parameters)
{
    double res = 0;

    double alphaj, betaj;
    auto alpha_start = parameters.begin();
    auto split = parameters.begin() + order;
    auto beta_end = parameters.end();
    // for (unsigned j = 0; j < samples.size(); ++j) {
    // for (const auto& s : samples) {
    for (auto it = samples.begin(); it != samples.begin() + 25000; ++it) {
        const auto& s = *it;
        // alpha(lambda_j)
        alphaj = Polyn(alpha_start, split, s.nu);
        betaj = Polyn(split, beta_end, s.nu);

        if (alphaj <= alphaMin || betaj <= betaMin)
            return -std::numeric_limits<double>::infinity();

        // cout << "alphaj " << alphaj << ", betaj "<< betaj << endl;

        res += alphaj * log(betaj) - lgamma(alphaj) + (alphaj - 1) * log(s.en) - betaj * s.en;
        if (!std::isfinite(res)) {
            cout << "res not finite! " << alphaj << ", " << betaj << endl;
            throw 2;
        }
    }
    // cout << "likelihood " << res << endl;

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

     static const double invalid = -std::numeric_limits<double>::infinity();
     static const double valid = 0;

     const double alphaMin = MinPolyn(parameters.begin(), parameters.begin() + order);
     if (alphaMin <= this->alphaMin)
         return invalid;

     const double betaMin = MinPolyn(parameters.begin() + order, parameters.end());
     if (betaMin <= this->betaMin)
         return invalid;

     // default: uniform prior
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
