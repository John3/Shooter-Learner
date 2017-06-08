import sys
import parameter_config as cfg

with open(sys.argv[2],'w') as hist:
    #size = 0
    #with open(sys.argv[1]) as obs:
    #    for c in obs:
    #        size = size +1
    for i in range(9):
        q = 0
        with open(sys.argv[1]) as obs:
            for c in obs:
                if c.rstrip() == str(i):
                    q = q + 1
        hist.write("%s %s\n"% (cfg.action_to_string[i], q))
