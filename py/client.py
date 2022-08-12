from webbrowser import get
import python_gen as gpuprofiling
import argparse
def getProfilingResponseFromPB(s:str)->gpuprofiling.GPUProfilingResponse:
    with open(s, "rb") as f:
        return gpuprofiling.GPUProfilingResponse.FromString(f.read())
def getProfilingResponseFromFile(s:str)->gpuprofiling.GPUProfilingResponse:
    return gpuprofiling.GPUProfilingResponse()
if __name__ =="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument('--addr',type=str,dest='addr',default='localhost:8886',help='the address to connect to')
    parser.add_argument('--duration',type=int,dest='duration',default=2000,help='PC sampling duration')
    parser.add_argument('--dumpfn',type=str,dest='dumpfn',help='the dumped profile filename')
    parser.add_argument('--pbfn',type=str,dest='pbfn',help='the dumped pb profile filename')
    parser.add_argument('--prune',dest='prune',action='store_true',help='prune the cpuCCTS')
    args=parser.parse_args()
    r=gpuprofiling.GPUProfilingResponse()
    if args.pbfn is not None:
        r=getProfilingResponseFromPB(args.pbfn)
    elif args.dumpfn is not None:
        r=getProfilingResponseFromFile(args.dumpfn)
    else:
        # TODO initiate grpc call
        pass

    if args.prune is not None:
        #TODO prune cct
        pass
    #/root/GVProf-samples/pytorch/profiling_response.pb.gz
    print(r)