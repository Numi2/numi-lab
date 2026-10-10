def forward(points,local,weights,row,tissue,poses):
    f=np.float32; out=np.zeros_like(points,dtype=np.float32)
    for vi in range(len(points)):
        accum=np.zeros(3,dtype=np.float32)
        for slot in range(4):
            w=f(weights[vi,slot])
            if w<=f(0): continue
            li=int(local[vi,slot]); require(li<row["bc"],"candidate local binding out of range")
            binding=tissue["bindings"][row["fb"]+li]; core=int(binding["core"])
            require(core in poses,"missing body pose "+str(core))
            val=binding["value"]; rr=rotate32(val[3:7],points[vi])
            loc=np.asarray([f(val[j]+f(f(rr[j])*val[7])) for j in range(3)],dtype=np.float32)
            bp,bq=poses[core]; wr=rotate32(bq,loc)
            world=np.asarray([f(bp[j]+wr[j]) for j in range(3)],dtype=np.float32)
            accum=np.asarray([f(accum[j]+f(w*world[j])) for j in range(3)],dtype=np.float32)
        out[vi]=accum
    return out
