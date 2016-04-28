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

void extractBinX(double numin, double numax, const std::string& outFile, unsigned maxElements)
{
    std::ofstream file(outFile);
    for (unsigned i = 0; i < 250; ++i) {
        Tardis m("harr", "../../posterior/real_tardis_250.h5", i, maxElements);
        auto res = m.SumX(numin, numax);
        file << std::get<0>(res) << '\t' << std::get<1>(res) << endl;
    }
}

int main(int argc, char* argv[])
{
    // static const double numin = 0.018;
    // static const double numax = 0.0185;
    static const double numin = 0.01;
    static const double numax = 0.05;
    constexpr unsigned maxElements = 20000;

#if 1
    extractBinX(numin, numax, "X.out", maxElements);
    return 0;
#endif

    if (argc < 2) {
        cerr << "Provide run number 0..249!" << endl;
        return 1;
    }
    int run = stoi(argv[1]);

    // set nicer style for drawing than the ROOT default
    BCAux::SetStyle();

    // open log file
    BCLog::OpenLog("log.txt", BCLog::detail, BCLog::debug);

    // create new Tardis object
    Tardis m("tardis", "../../posterior/real_tardis_250.h5", run, maxElements);

    static const double mean = 0.02;
    static const double nu = (numax - numin) / 2.0;
    auto res = m.SumX(numin, numax);
    const unsigned n = std::get<0>(res);

    //    double X = std::get<1>(res);


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
#if 1
    m.PreparePrediction();

    std::ofstream file(std::string("out") + to_string(run) + ".txt");

    // P(0) = 0 but leads to numerical break down
    file << 0. << '\t' << 0. << endl;

    constexpr auto K = 100;
    auto Xmax = 3.0 * n * mean;
    for (auto i = 1; i <= K; ++i) {
        const double X = double(i) / K * Xmax;
        const double P = m.PredictSmall(n, X, nu, mean, 0.05);
//        const double P = m.PredictMedium(n, X, nu);
//        const double P = m.PredictVeryLarge(n, X, nu);
        file << X << '\t' << P << endl;
    }
#endif
#if 0
    m.PreparePrediction();
    m.PredictMedium(n, n * mean, nu);

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
#if 1
    // clumsy way to set only the parameters but not the observables at the mode
    Tardis::Vec harr(m.GetBestFitParameters().begin(), m.GetBestFitParameters().begin() + m.GetNParameters());
    // BCLog::OutDetail(Form("length %u", harr.size()));
    // BCLog::OutDetail(Form("npar %u", m.GetNParameters()));
//    m.SetInitialPositions(std::vector<double>(m.GetBestFitParameters().begin(), m.GetBestFitParameters().begin() + m.GetNParameters()));
    m.SetInitialPositions(std::vector<double>(m.GetMinuit().X(), m.GetMinuit().X() + m.GetNParameters()));
//    return 0;
#endif
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
