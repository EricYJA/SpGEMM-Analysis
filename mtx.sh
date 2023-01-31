listv="mc2depi.mtx pwtk.mtx amazon03.mtx cant.mtx consph.mtx pdb1HYS.mtx bcsstk17.mtx shipsec1.mtx rma10.mtx cop20k_A.mtx mac_econ.mtx scircuit.mtx cit-HepP.mtx p2p-Gnut.mtx soc-Epin.mtx soc-sign.mtx sx-matho.mtx email-Eu.mtx sx-askub.mtx geomean.mtx mc2de.mtx pwtk_.mtx amazo.mtx cant_.mtx consp.mtx pdb1H.mtx bcsst.mtx ships.mtx mac_e.mtx scirc.mtx cit-H.mtx p2p-G.mtx soc-E.mtx soc-s.mtx sx-ma.mtx sx-as.mtx geome.mtx cop20k.mtx mathoverflow.mtx cit.mtx soc-Epinions1.mtx p2p.mtx sign-epinions.mtx email-EuAll.mtx amazon0302.mtx cop20.mtx signmatho.mtx email.mtx enron.mtx askub.mtx"
for v in $listv; do
# /data/toodemuy/data/AAT
# /data/toodemuy/datasets/floridaMatrices
# /data/toodemuy/datasets/SNAP
 find /data/suitesparse_dataset/MM -name "$v" | xargs cp -t /home/ychenfei/research/sparse_matrix_perf_analysis/spgemm_dataflow_analysis/test_mtx2
done