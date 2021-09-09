using MPI
using Plots
MPI.Init()

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
size = MPI.Comm_size(comm)

dst = mod(rank+1, size)
src = mod(rank-1, size)

bw  = Array{Float64}(undef, 12)
print("$rank: Generating Data\n")
for n in 1:12
    N = 2^n
    send_msg = Array{Int8}(undef,N)
    recv_msg = Array{Int8}(undef,N)
    fill!(send_msg, Int8(rank))
    rreq = MPI.Irecv!(recv_msg, src, src+32, comm)
    # print("$rank: Sending   $rank -> $dst\n")
    t  = @elapsed (MPI.Send(send_msg, dst, rank+32, comm))
    stats = MPI.Waitall!([rreq])
    # print("$rank: Received $src -> $rank\n")
    val = (n/t)*10e-6
    bw[n] = val
end
print("$rank: Data Generated, Starting plot\n")
# print("$rank: $bw")

msize = [2^n for n in 1:12]

plot(msize, bw, xlabel="Message Size(bytes)", ylabel="Bandwith(MB/s)", title="Process $rank", size=(600, 600), label=false)

png("D:\\JuliaDev\\Scripts\\Plots\\mpass$rank")

MPI.Barrier(comm)
