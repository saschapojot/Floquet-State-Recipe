import floquet_obcx
from datetime import datetime
def runObcx():
    start = datetime.now()
    obcx = floquet_obcx.obcxSolver()
    obcx.calculatePhases()
    obcx.separateStates()
    obcx.plotSpectrum()
    obcx.plotEdgeStates()
    end = datetime.now()
    print("Time: ", end - start)


def main():
    runObcx()


if __name__ == '__main__':
    main()
