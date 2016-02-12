// ***************************************************************
// This file was created using the bat-project script
// for project Tardis.
// bat-project is part of Bayesian Analysis Toolkit (BAT).
// BAT can be downloaded from http://mpp.mpg.de/bat
// ***************************************************************

#include <BAT/BCLog.h>
#include <BAT/BCAux.h>

#include "Tardis.h"

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
    m.SetMinimumEfficiency(0.08);
    m.SetMaximumEfficiency(0.25);
    m.SetNChains(2);
    m.SetRValueParametersCriterion(1.2);

    m.SetNIterationsRun(3000);

    BCLog::OutSummary("Test model created");

    //////////////////////////////
    // perform your analysis here

    // Normalize the posterior by integrating it over the full par. space
    // m.Normalize();

    // run MCMC, marginalizing posterior
    m.MarginalizeAll(BCIntegrate::kMargMetropolis);

    // run mode finding; by default using Minuit
    m.FindMode(m.GetBestFitParameters());

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
}
