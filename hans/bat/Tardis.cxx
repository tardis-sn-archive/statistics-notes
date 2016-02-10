// ***************************************************************
// This file was created using the bat-project script.
// bat-project is part of Bayesian Analysis Toolkit (BAT).
// BAT can be downloaded from http://mpp.mpg.de/bat
// ***************************************************************

#include "Tardis.h"

#include <BAT/BCMath.h>

#include <H5Cpp.h>

#include <cassert>
#include <cmath>
#include <iostream>
#include <numeric>

using std::cout;
using std::endl;
using namespace H5;

// ---------------------------------------------------------
Tardis::Tardis(const std::string& name)
    : BCModel(name),
      order(3)
{
    // Define parameters here in the constructor. For example:
    // AddParameter("mu",-2,1,"#mu");
    // And set priors, if using built-in priors. For example:
    // GetParamater("mu").SetPrior(new BCPriorGaus(-1, 0.25));

    AddParameter("alpha0",  1, 2, "#alpha_0");
    AddParameter("alpha1", -1, 1, "#alpha_1"); GetParameter(1).Fix(0);
    AddParameter("alpha2", -1, 1, "#alpha_2"); GetParameter(2).Fix(0);

    AddParameter("beta0",  0, 2, "#beta_0");
    AddParameter("beta1", -1, 1, "#beta_1"); GetParameter(4).Fix(0);
    AddParameter("beta2", -1, 1, "#beta_2"); GetParameter(5).Fix(0);

    ReadData("../../posterior/real_tardis_250.h5", "energies", energies);
    ReadData("../../posterior/real_tardis_250.h5", "nus", nus);

    assert(energies.size() == nus.size());
}

// ---------------------------------------------------------
Tardis::~Tardis()
{
}

void Tardis::ReadData(const std::string& fileName, const std::string& dataSet, std::vector<double>& buffer)
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
    int ndims = dataspace.getSimpleExtentDims( dims_out, NULL);
//    cout << "rank " << ndims << ", dimensions " <<
//        (unsigned long)(dims_out[0]) << " x " <<
//        (unsigned long)(dims_out[1]) << endl;

    // read one row from large matrix into 1D array
    static const int rankOut = 1;
    hsize_t N = dims_out[1];
    std::array<hsize_t, 2> offsetIn = {run, 0};
    std::array<hsize_t, 2> countIn = {1, N};
    dataspace.selectHyperslab(H5S_SELECT_SET, &countIn[0], &offsetIn[0]);

    // reserve memory to hold the data
    buffer.reserve(N);

    // define memory buffer layout
    DataSpace memspace(rankOut, &N);

    // where to write into the buffer: start at beginning and write up to the end
    hsize_t offset = 0;
    hsize_t count = N;
    memspace.selectHyperslab(H5S_SELECT_SET, &count, &offset);

    // read and possibly convert the data
    dataset.read(&buffer[0], PredType::NATIVE_DOUBLE, memspace, dataspace);
}

// ---------------------------------------------------------
double Tardis::LogLikelihood(const std::vector<double>& parameters)
{
    double res = 0;

    double alphaj, betaj;
    for (unsigned j = 0; j < nus.size(); ++j) {
        // alpha(lambda_j)
        alphaj = polyn(parameters.begin(), parameters.begin() + order, nus[j]);
        betaj = polyn(parameters.begin() + order, parameters.end(), nus[j]);

        res += alphaj * log(betaj) - lgamma(alphaj) + (alphaj - 1) * log(energies[j]) - betaj * energies[j];
    }

    return -1;
}

// ---------------------------------------------------------
 double Tardis::LogAPrioriProbability(const std::vector<double> & parameters) {
     // This returns the log of the prior probability for the parameters
     // If you use built-in 1D priors, don't uncomment this function.

     // alpha > 1
     if (std::accumulate(parameters.begin(), parameters.begin() + order, 0) <= 1.0)
         return -std::numeric_limits<double>::infinity();
     // beta > 0
     if (std::accumulate(parameters.begin() + order, parameters.end(), 0) <= 0.0)
         return -std::numeric_limits<double>::infinity();

     // default: uniform prior
     return 0;
}

double Tardis::polyn(Vec::const_iterator first, Vec::const_iterator last, const double& lambda)
{
    double res = 0;
    double power = 1;
    for (; first != last; ++first) {
        res += (*first) * power;
        power *= lambda;
    }
    return res;
}
