// ***************************************************************
// This file was created using the bat-project script
// for project Tardis.
// bat-project is part of Bayesian Analysis Toolkit (BAT).
// BAT can be downloaded from http://mpp.mpg.de/bat
// ***************************************************************

#include <BAT/BCLog.h>
#include <BAT/BCAux.h>

#include "Tardis.h"

using namespace std;

int main()
{
    // set nicer style for drawing than the ROOT default
    BCAux::SetStyle();

    // open log file
    BCLog::OpenLog("log.txt", BCLog::debug, BCLog::debug);

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

    m.PreparePrediction();
    static const unsigned n = 8;
    static const double mean = 0.02;
    static const double nu = 0.0125;
    cout << "Predict " << m.PredictSmall(n, n * mean, nu, 1e-3) << endl;

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
