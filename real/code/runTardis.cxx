#include "Tardis.h"

#include <fstream>
#include <iostream>

using namespace std;
using namespace ROOT::Minuit2;

bool file_exists(const std::string& filename)
{
    std::ifstream infile(filename);
    return infile.good();
}

array<double, 2> extractBinX(double numin, double numax, const std::string& filename, unsigned maxElements)
{
    double min = std::numeric_limits<double>::infinity();
    double max = -min;

    if (file_exists(filename)) {
        /* read it in */
        std::ifstream infile(filename);
        int ninbin, totaln;
        double X;
        while (infile >> ninbin >> X >> totaln) {
            min = std::min(min, X);
            max = std::max(max, X);
        }
    } else {
        /* compute numbers and write to file */
        std::ofstream file(filename);
        for (unsigned i = 0; i < 250; ++i) {
            Tardis m("../../posterior/real_tardis_250.h5", i, maxElements);
            auto res = m.SumX(numin, numax);
            const auto& X = std::get<1>(res);
            min = std::min(min, X);
            max = std::max(max, X);
            file << std::get<0>(res) << '\t' << X
                    << '\t' << m.Nsamples() << endl;
        }
    }

    return array<double, 2>{{min, max}};
}

int main(int argc, char* argv[])
{
    constexpr double numin = 0.1;
    constexpr double numax = 0.2;
    constexpr unsigned maxElements = 0; // 20000;

    std::string outprefix("X" + std::to_string(numin) + "-" + std::to_string(numax));
    auto minmaxX = extractBinX(numin, numax, outprefix + "_replica.out", maxElements);
    cout << "min " << get<0>(minmaxX)<< ", max " << get<1>(minmaxX)<< endl;

    if (argc < 2) {
        cerr << "Provide run number 0..249!" << endl;
        return 1;
    }
    int run = stoi(argv[1]);

    // create new Tardis object
    Tardis m("../../posterior/real_tardis_250.h5", run, maxElements);

    constexpr double nu = (numax - numin) / 2.0;
    auto res = m.SumX(numin, numax);
    const unsigned n = std::get<0>(res);

//    m.fitnb();
//    m.FixPredicted(Tardis::Target::NBGamma, n, get<1>(res), numin + 0.5 * (numax - numin));
//    FunctionMinimum minimum = m.minimizeMinuit();

    //    double X = std::get<1>(res);


//    double meanX = m.SumX(0.018, 0.0185) / n;

    /*
     * Surprising behavior: passing the mean reduces the total number
     * of calls to minuit from 549 to 472 but the number of calls to
     * the likelihood increases from 36266 to 45117.  The reason may
     * be that jumping from left side to right side of mode in N
     * changes the parameters a lot so minuit has to search longer
     * based on the previous point. When the search is only in one
     * direction, the modes don't change very much.
     *
     * Running minuit from the last mode in that search direction gives the best overall result with 33749 calls but more debug output.
     *
     * Hyperthreading gives an improvement of ~15 - 20 % for the single likelihood
     */
#if 1
//    auto mode = m.PreparePrediction();

    auto mode = gsl_vector_alloc(5);
//    std::vector<double> vecmode {1.69318195, -0.2316184245, 0.06850367375, 78.41164031, -14.9692865};
    std::array<double, 5> vecmode {{ 1.601271636, -0.2258877812, 0.07348750746, 75.14014599, -14.30604913}};
    std::copy(vecmode.begin(), vecmode.end(), mode->data);

    std::ofstream file(outprefix + "_run" + to_string(run) + ".out");

    // P(0) = 0 but leads to numerical break down
    file << 0. << '\t' << 0. << endl;

    /* use range of values to define where prediction is needed
     *
     * Go from max(0, Xmin - (Xmax-xmin)/2
     *  */

    // number of points
    constexpr auto K = 1;
    const auto DX = (get<1>(minmaxX) - get<0>(minmaxX));
    const auto extra = 0.5;
    const auto Xmin = max(0.0, get<0>(minmaxX) - extra * DX);
    const auto Xmax = get<1>(minmaxX) + extra * DX;
    const auto dx = (Xmax - Xmin) / K;
    for (auto i = 1; i <= K; ++i) {
        const double X = Xmin + i * dx;
//        const double P = m.PredictSmall(n, X, nu, m.mean(), 0.01);
        const double P = m.PredictMedium(mode, n, X, nu);
//        const double P = m.PredictVeryLarge(n, X, nu);
        file << X << '\t' << P << endl;
    }
#endif
}

// Local Variables:
// compile-command: "make && ./runTardis"
// End:
