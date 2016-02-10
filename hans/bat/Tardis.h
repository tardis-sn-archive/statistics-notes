// ***************************************************************
// This file was created using the bat-project script.
// bat-project is part of Bayesian Analysis Toolkit (BAT).
// BAT can be downloaded from http://mpp.mpg.de/bat
// ***************************************************************

#ifndef __BAT__TARDIS__H
#define __BAT__TARDIS__H

#include <BAT/BCModel.h>

#include <string>
#include <vector>

// This is a Tardis header file.
// Model source code is located in file Tardis/Tardis.cxx

// ---------------------------------------------------------
class Tardis : public BCModel
{

public:
    using Vec = std::vector<double>;

    // Constructor
    Tardis(const std::string& name);

    // Destructor
    ~Tardis();

    void ReadData(const std::string& fileName, const std::string& dataSet, Vec& buffer);

    // Overload LogLikelihood to implement model
    double LogLikelihood(const std::vector<double>& parameters);

    // Overload LogAprioriProbability if not using built-in 1D priors
    double LogAPrioriProbability(const std::vector<double> & parameters);

private:
    double polyn(Vec::const_iterator begin, Vec::const_iterator end, const double& lambda);

    std::vector<double> energies, nus;

    const unsigned order;
};
// ---------------------------------------------------------

#endif
