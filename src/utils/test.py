from turbomoleOutputProcessing import turbomoleOutputProcessing as top


if __name__ == '__main__':
    filename = "/data/scc/matthias.blaschke/eigenchannel_test/transport/Trans_prob.dat"
    #a = top.read_hessian(filename, 31)
    filename = "/data/scc/matthias.blaschke/eigenchannel_test/transport/gammaL.dat"
    gammaL = top.read_hessian(filename, 31)
    filename = "/data/scc/matthias.blaschke/eigenchannel_test/transport/grcc.dat"
    #grcc = top.read_hessian(filename, 31)
    #print(a)