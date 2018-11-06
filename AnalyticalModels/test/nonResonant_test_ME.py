#! /usr/bin/env python
# Analytical reweighting implementation for H->4b
# This file is part of https://github.com/cms-hh/HHStatAnalysis.
# python nonResonant_test_NLO.py  --kl 1 --kt 1
# python nonResonant_test_NLO.py  --kl 0.0001 --kt 0.0001 --c2 -3.0 --cg 0.0 --c2g -1.5
# compiling
from optparse import OptionParser
import ROOT
import numpy as np
import os
from HHStatAnalysis.AnalyticalModels.NonResonantModel import NonResonantModel


parser = OptionParser()
parser.add_option("--kl", type="float", dest="kll", help="Multiplicative factor in the H trilinear wrt to SM")
parser.add_option("--kt", type="float", dest="ktt", help="Multiplicative factor in the H top Yukawa wrt to SM")

parser.add_option("--c2", type="float", dest="c22", help="ttHH with triangle loop", default=0)
parser.add_option("--cg", type="float", dest="cgg", help="HGG contact", default=0)
parser.add_option("--c2g", type="float", dest="c2gg", help="HHGG contact", default=0)
parser.add_option("--energy", type="int", dest="energy", help="LHC energy", default=13)
parser.add_option("--doPlot", action='store_true', default=False, dest='doPlot',
                    help='calculate the limit in the benchmark poin specified')

(options, args) = parser.parse_args()
print " "
kl = options.kll
kt = options.ktt
c2 = options.c22
cg = options.cgg
c2g = options.c2gg

print "Weights calculated from the 12 benchmarks defined in 1507.02245v4 (JHEP version) each one with 100k events "
###########################################################
# read events and apply weight
###########################################################
def main():
  model = NonResonantModel()
  model.ReadCoefficients2(options.energy, "/afs/cern.ch/work/a/acarvalh/CMSSW_8_1_0/")
  effSumV0 = 1.0
  Cnorm = 1.0

  histfilename="../../../Analysis/Support/NonResonant/HistSum2D_4b_rebin_SimCostHH_19-4.root"
  ### need to do one with SM+bench2 with the binning required
  histtitle= "SumV0_AnalyticalBinExtSimCostHH" #
  fileHH=ROOT.TFile(histfilename)
  sumHAnalyticalBin = fileHH.Get(histtitle)
  kl = 1.0
  kt = 1.0
  c2 = 0.0
  cg = 0.0
  c2g = 0.0
  calcSumOfWeights = model.getNormalization2(kl, kt,c2,cg,c2g) #/model.neventHist

  inputSamples = "/eos/cms/store/user/acarvalh/asciiHH_tofit/GF_HH_BSM/events_SumV0.root" ## user input
  ### the user need to make a tree with GenHHCost / Genmhh
  ### with the events of the input samples that are being reweighted === without cuts!!!
  inputSamplestree = "treeout"
  ## name the branches as bellow:
  ## branchGenmhh = "Genmhh"
  ## branchGenHHCost = "GenHHCost"
  if not os.path.isfile("HistoInputEvents.root") :
      print ("creating histo for the denominator of the weights")
      model.getInputInHisto(inputSamples, inputSamplestree)
  fileAllHisto = ROOT.TFile("HistoInputEvents.root")
  inputSamplesHisto = fileAllHisto.Get("EventsSum")
  print ("there are ~", inputSamplesHisto.Integral(), "input events without cuts")
  ####
  weight = 0
  fileAll = ROOT.TFile(inputSamples)
  # it can be a tree with or without cuts, provided that the version without cuts coincide with inputSamples
  tree = fileAll.Get(inputSamplestree)
  print ("there are ~", tree.GetEntries(), "input events to reweight (with or without cuts)")
  nev = tree.GetEntries()
  # declare the histograms
  CalcMhh = np.zeros((nev))
  CalcCost = np.zeros((nev))
  CalcWeight = np.zeros((nev))
  countevent = 0
  counteventNonZero = 0
  sumWeight = 0
  for ee, event in enumerate(tree) :
      if ee % 200000 == 0 : print ("processed", ee)
      mhh = event.Genmhh+0.01
      cost = abs(event.GenHHCost)
      weight = model.getScaleFactor2(mhh, cost, kl, kt,c2,cg,c2g, inputSamplesHisto)/calcSumOfWeights
      #weight = model.getScaleFactor_fromSMonly(mhh, cost, kl, kt,c2,cg,c2g)/calcSumOfWeights
      #############################################
      # fill histograms to test
      #############################################
      CalcMhh[countevent] = float(mhh)
      CalcCost[countevent] = float(cost)
      if weight < 0 : weight = 0.0
      else : counteventNonZero+=1
      CalcWeight[countevent] = weight
      countevent+=1
      sumWeight+=weight
  print "plotted histogram reweighted from ",counteventNonZero," events, ", float(100*(nev-counteventNonZero)/nev)," % of the events was lost in empty bins in SM simulation"
  print "sum of weights (== signal efficiency)",sumWeight

  ###############################################
  # Draw test histos
  ###############################################
  drawtest =0
  nevtest=50000
  if kl == 1 and kt == 1 and c2 ==0 and cg == 0 and c2g ==0 :
     filne = "/eos/cms/store/user/acarvalh/asciiHH_tofit/GF_HH_BSM/GF_HH_0.lhe.decayed"    # 0 is SM
     nevtest = 100000
     drawtest = 1
  # BSM events
  pathBSMtest="/eos/cms/store/user/acarvalh/asciiHH_tofit/GF_HH_toRecursive/" # events of file to superimpose a test
  # see the translation of coefficients for this last on: If you make this script smarter (to only read files we ask to test) you can implement more
  # https://github.com/acarvalh/generateHH/blob/master/fit_GF_HH_lhe/tableToFitA3andA7.txt
  if kl == -10 and kt == 0.5 and c2 ==0 and cg == 0 and c2g ==0 :
     drawtest =1
     filne = pathBSMtest+"GF_HH_42.lhe.decayed"
  if kl == 0.0001 and kt == 2.25 and c2 ==0 and cg == 0 and c2g ==0  :
     drawtest =1
     filne = pathBSMtest+"GF_HH_9.lhe.decayed"
  if kl == 2.5 and kt == 1.0 and c2 ==0 and cg == 0 and c2g ==0  :
     drawtest =1
     filne = pathBSMtest+"GF_HH_60.lhe.decayed"
  klJHEP=[1.0, 7.5,  1.0,  1.0,  -3.5, 1.0, 2.4, 5.0, 15.0, 1.0, 10.0, 2.4, 15.0]
  ktJHEP=[1.0, 1.0,  1.0,  1.0,  1.5,  1.0, 1.0, 1.0, 1.0,  1.0, 1.5,  1.0, 1.0]
  c2JHEP=[0.0, -1.0, 0.5, -1.5, -3.0,  0.0, 0.0, 0.0, 0.0,  1.0, -1.0, 0.0, 1.0]
  cgJHEP=[0.0, 0.0, 0.6,  0.0, 0.0,   0.8, 0.2, 0.2, -1.0, -0.6, 0.0, 1.0, 0.0]
  c2gJHEP=[0.0, 0.0, 1.0, -0.8, 0.0, -1.0, -0.2,-0.2,  1.0,  0.6, 0.0, -1.0, 0.0]
  # python nonResonant_test_NLO.py  --kl 7.5 --kt 1 --c2 -1
  # python nonResonant_test_NLO.py  --kl 1.0 --kt 1.0 --c2 0.5 --cg -0.8 --c2g 0.6
  # python nonResonant_test_NLO.py  --kl 1.0 --kt 1.0 --c2 -1.5 --cg 0.0 --c2g -0.8
  # python nonResonant_test_NLO.py  --kl 1.0 --kt 1.0 --c2 0.0 --cg 0.8 --c2g -1.0
  # python nonResonant_test_NLO.py  --kl 1.0 --kt 1.0 --c2 1.0 --cg -0.6 --c2g 0.6
  # python nonResonant_test_NLO.py  --kl -3.5 --kt 1.5 --c2 -3.0
  for sam in range(0,13):
    #print (sam, ktJHEP[sam] , kt , klJHEP[sam] , c2 ,c2JHEP[sam] , cg , cgJHEP[sam] , c2g , c2gJHEP[sam])
    if kl == klJHEP[sam] and kt == ktJHEP[sam] and c2 ==c2JHEP[sam] and cg == cgJHEP[sam] and c2g ==c2gJHEP[sam] :
       print ("It is the shape benchmark:", sam)
       filne="/eos/cms/store/user/acarvalh/asciiHH_tofit/GF_HH_BSM/GF_HH_"+str(sam)+".lhe.decayed"
       nevtest=100000
       drawtest = sam
  # BSM events
  pathBSMtest="/eos/cms/store/user/acarvalh/asciiHH_tofit/GF_HH_toRecursive/" # events of file to superimpose a test
  # see the translation of coefficients for this last on: If you make this script smarter (to only read files we ask to test) you can implement more
  # https://github.com/acarvalh/generateHH/blob/master/fit_GF_HH_lhe/tableToFitA3andA7.txt
  if kl == -10 and kt == 0.5 and c2 ==0 and cg == 0 and c2g ==0 :
     drawtest =-1
     filne = pathBSMtest+"GF_HH_42.lhe.decayed"
  if kl == 0.0001 and kt == 2.25 and c2 ==0 and cg == 0 and c2g ==0  :
     drawtest =-1
     filne = pathBSMtest+"GF_HH_9.lhe.decayed"
  if kl == 0.0001 and kt == 1.0 and c2 ==0 and cg == 0 and c2g ==0  :
     drawtest =-1
     filne = pathBSMtest+"GF_HH_4.lhe.decayed"
  if kl == 2.5 and kt == 1.0 and c2 ==0 and cg == 0 and c2g ==0  :
     drawtest =-1
     filne = pathBSMtest+"GF_HH_60.lhe.decayed"
  if kl == -15 and kt == 0.5 and c2 ==0 and cg == 0 and c2g ==0  :
     drawtest =-1
     filne = pathBSMtest+"GF_HH_40.lhe.decayed"
  if kl == 5 and kt == 1.5 and c2 ==0 and cg == 0 and c2g ==0  :
     drawtest =-1
     filne = pathBSMtest+"GF_HH_74.lhe.decayed"
  if kl == 7.5 and kt == 2.0 and c2 ==0 and cg == 0 and c2g ==0  :
     drawtest =-1
     filne = pathBSMtest+"GF_HH_88.lhe.decayed"
  if kl == 0.0001 and kt == 0.0001 and c2 ==-3.0 and cg == 0.0 and c2g ==-1.5  :
     drawtest =-1
     filne = pathBSMtest+"GF_HH_281.lhe.decayed"
  if kl == -10 and kt == 0.0 and c2 ==1.0 and cg == 1.0 and c2g ==1.0  :
     drawtest =-1
     filne = pathBSMtest+"GF_HH_280.lhe.decayed"
  ########################################################################################
  CalcMhhTest = np.zeros((nevtest))
  CalcCostTest = np.zeros((nevtest))
  if options.doPlot :
    print "draw plain histogram to test"
    model.LoadTestEvents(CalcMhhTest,CalcCostTest,filne)
    model.plotting(kl,kt,c2,cg,c2g,CalcMhh,CalcCost,CalcWeight,CalcMhhTest,CalcCostTest,drawtest)

##########################################
if __name__ == "__main__":
   main()
