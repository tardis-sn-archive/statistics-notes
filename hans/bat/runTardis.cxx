// ***************************************************************
// This file was created using the bat-project script
// for project Tardis.
// bat-project is part of Bayesian Analysis Toolkit (BAT).
// BAT can be downloaded from http://mpp.mpg.de/bat
// ***************************************************************

#include "Tardis.h"

#include <BAT/BCLog.h>
#include <BAT/BCAux.h>

#include <fstream>

using namespace std;

int main()
{
    // set nicer style for drawing than the ROOT default
    BCAux::SetStyle();

    // open log file
    BCLog::OpenLog("log.txt", BCLog::detail, BCLog::detail);

    // create new Tardis object
    Tardis m("tardis");

    // set precision
    m.SetPrecision(BCEngineMCMC::kLow);
    // m.SetProposeMultivariate(false);
    m.SetMinimumEfficiency(0.15);
    m.SetMaximumEfficiency(0.40);
    m.SetNChains(5);
    m.SetRValueParametersCriterion(1.15);
//    m.SetInitialPositionScheme(BCEngineMCMC::kInitRandomUniform);
    m.SetNIterationsPreRunMax(20000);
   m.SetProposalFunctionDof(-1);

    m.SetNIterationsRun(3000);

    BCLog::OutSummary("Test model created");

    static const unsigned n = 8;
    static const double mean = 0.02;
    static const double nu = 0.0125;

    // find bin edge


//    double meanX = m.SumX(0.018, 0.0185) / n;

    /*
     * Surprising behavior: passing the mean reduces the total number
     * of calls to minuit from 549 to 472 but the number of calls to
     * the likelihood increases from 36266 to 45117.  The reason may
     * be that jumping from left side to right side of mode in N
     * changes the parameters a lot so minuit has to search longer
     * based on the previous point. When the search is only in on
     * direction, the modes don't change very much.
     *
     * Running minuit from the last mode in that search direction gives the best overall result with 33749 calls but more debug output.
     *
     * Hyperthreading gives an improvement of ~15 - 20 % for the single likelihood
     */

    m.PreparePrediction();
    std::ofstream file("out.txt");

    for (unsigned i = 1; i <= 4 * n; ++i) {
        const double X = mean * i;
        file << X << '\t' << m.PredictSmall(n, X, nu, mean, 5e-3) << endl;
    }

#if 0

    // clumsy way to set only the parameters but not the observables at the mode
    Tardis::Vec harr(m.GetBestFitParameters().begin(), m.GetBestFitParameters().begin() + m.GetNParameters());
    // BCLog::OutDetail(Form("length %u", harr.size()));
    // BCLog::OutDetail(Form("npar %u", m.GetNParameters()));
    m.SetInitialPositions(std::vector<double>(m.GetBestFitParameters().begin(), m.GetBestFitParameters().begin() + m.GetNParameters()));
//    return 0;

//    // run MCMC, marginalizing posterior
    m.MarginalizeAll(BCIntegrate::kMargMetropolis);
//
//    // run mode finding; by default using Minuit

    // draw all marginalized distributions into a PDF file
    m.PrintAllMarginalized(m.GetSafeName() + "_plots.pdf");

    // print summary plots
    // m.PrintParameterPlot(m.GetSafeName() + "_parameters.pdf");
    // m.PrintCorrelationPlot(m.GetSafeName() + "_correlation.pdf");
    // m.PrintCorrelationMatrix(m.GetSafeName() + "_correlationMatrix.pdf");
    // m.PrintKnowledgeUpdatePlots(m.GetSafeName() + "_update.pdf");

    // print results of the analysis into a text file
    m.PrintSummary();

    // close log file
    BCLog::OutSummary("Exiting");
    BCLog::CloseLog();

    return 0;
#endif
}

// Local Variables:
// compile-command: "make && ./runTardis"
// End:
