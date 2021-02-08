options(stringsAsFactors = F)
d = read.table('hw6.counts.txt', check.names = F)
d = as.matrix(d)
dim(d)

d = d[apply(d,1,mean)>=10,]
dim(d)
d = sweep(d,2,apply(d,2,sum),'/')
meta = strsplit(colnames(d),'_',fixed = TRUE)
meta = do.call(rbind, meta) 
m = as.data.frame(meta) 
colnames(m) = c('tissue', 'age')
m$age = as.numeric(m$age) 

age = m$age^0.25
age_2 = m$age^0.5
model = y ~ m$tissue + age +age_2 + m$tissue:age + m$tissue:age_2


pv.lm = matrix(ncol = 5, nrow = nrow(d))

for (i in 1:nrow(d)) {
  y = d[i,]
  reg=lm(model)
  pv.lm[i,] = anova(reg)[1:5,5]
}
colnames(pv.lm) = c('tissue', 'age', 'age_2', 'tissue:age', 'tissue:age_2')
pv.lm.bh = apply(pv.lm, 2, p.adjust, m = 'BH') #bonferroni correction

sg.tissue = sum(pv.lm.bh[,1] <0.05) # # significant tissue
sg.age = sum(pv.lm.bh[,2] <0.05) # # significant age
sg.age_2 = sum(pv.lm.bh[,3] <0.05) # # significant age_2
sg.age.tissue = sum(pv.lm.bh[,4] <0.05)# # significant age:tissue
sg.age_2.tissue = sum(pv.lm.bh[,5] <0.05)# # significant age_2:tissue

pv.thr = c()
for (i in 1:ncol(pv.lm.bh)){
  pv.th = max(pv.lm[pv.lm.bh[,i]<0.05,i])
  pv.thr = c(pv.thr, pv.th)
} 
pv.thr

pv.lm.perm = matrix(ncol = 5, nrow = nrow(d))
sgn.tissue.perm = c()
sgn.age.perm = c()
sgn.tissue.age.perm = c()

#permutation
for (j in 1:5) {
  order = sample(1:ncol(d))
  d_2 =d[,order]
  #permutation lm p-values
  for (i in 1:nrow(d)) {
    y = d_2[i,]
    reg=lm(model)
    pv.lm.perm[i,] = anova(reg)[1:5,5]
  }
  #pv.lm.bh.perm = apply(pv.lm.perm, 2, p.adjust, m = 'BH')
  
  sgn.t.p = sum(pv.lm.perm[,1] <pv.thr[1])
  #cat('\r','gene:', sgn.t)
  sgn.tissue.perm[j] = sgn.t.p
  sgn.a.p = sum(pv.lm.perm[,2] <pv.thr[2])
  sgn.age.perm[j] = sgn.a.p
  sgn.t.a.p = sum(pv.lm.perm[,4] <pv.thr[4])
  sgn.tissue.age.perm[j] = sgn.t.a.p
  
} # # significant tissue 0
fdr.tissue = mean(sgn.tissue.perm)/sg.tissue *100
fdr.age = mean(sgn.age.perm)/sg.age *100
fdr.tissue.age = mean(sgn.tissue.age.perm)/sg.age.tissue *100
