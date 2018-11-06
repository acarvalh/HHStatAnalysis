# Provides scale factors for event reweighting based on an analytical model.
# This file is part of https://github.com/cms-hh/HHStatAnalysis.
# compiling

import ROOT
import numpy as np
from array import array
import matplotlib
import matplotlib.pyplot as plt
import math
from root_numpy import tree2array

class NonResonantModel:
    def __init__(self):
        # read coefficients from the input file here
        # to store coefficients use self.
        self.NCostHHbin=4
        self.NMHHbin=55
        self.NCoef=15
        self.binGenMHH  = [ 250,260,270,280,290,300,310,320,330,340,
                               350,360,370,380,390,400,410,420,430,440,
                               450,460,470,480,490,
                               500,510,520,530,540,550,600,610,620,630,
                               640,650,660,670,680,690,700,750,800,850,
                               900,950,1000,1100,1200,1300,1400,1500.,1750,2000,50000]
        self.binGenCostS  = [ 0.0,0.4,0.6,0.8, 1.0 ] #
        self.effSM = np.zeros((self.NCostHHbin,self.NMHHbin))
        self.effSum = np.zeros((self.NCostHHbin,self.NMHHbin))
        self.MHH = np.zeros((self.NCostHHbin,self.NMHHbin))
        self.COSTS = np.zeros((self.NCostHHbin,self.NMHHbin))
        self.A =  [[[0 for MHHbin in range(self.NMHHbin)] for CostHHbin in range(self.NCostHHbin)] for coef in range(self.NCoef)]
        #self.A13tev = [2.09078, 10.1517, 0.282307, 0.101205, 1.33191, -8.51168, -1.37309, 2.82636, 1.45767, -4.91761, -0.675197, 1.86189, 0.321422, -0.836276, -0.568156]
        self.A13tev = [2.09078, 10.15, 0.282307, 0.101205, 1.33191, -8.51168, -1.37309, 2.83, 1.45767, -4.91761, -0.675197, 1.86189, 0.321422, -0.836276, -0.568156]
        self.A14tev = [2.100318379, 10.2, 0.287259045, 0.098882779, 1.321736614, -8.42431259, -1.388017366, 2.8, 0.518124457, -2.163473227, -0.550668596, 5.871490593, 0.296671491, -1.172793054, 0.653429812]
        self.Cnorm =0
        self.neventHist = 500000. #### for reweighting2
        self.NCoefNLO=24
        self.binGenMHHNLO  = [250, 270, 290, 310, 330, 350, 370, 390, 410, 430, 450,
        470, 490, 510, 530, 550, 570, 590, 610, 630, 650, 670, 690, 710, 730, 750, 770,
        790, 810, 830, 850, 870, 890, 910, 930, 950, 970, 990, 1010, 1030, 10000] # 41
        self.fileCoef = ROOT.TFile()
        self.fileCoef13TeV = 0
        self.fileCoef14TeV = 0
        self.AbinHist = []
        self.AbinTot = []
        self.ANLO =  [[0 for MHHbin in range(len(self.binGenMHHNLO)-1)] for coef in range(self.NCoefNLO)]
        self.A13tevNLO = [2.23389, 12.4598, 0.342248, 0.346822, 13.0087, -9.6455,
        -1.57553, 3.43849, 2.86694, 16.6912, -1.25293, -5.81216, 0.649714, 2.85933,
        3.14475, -0.00816241, 0.0208652, 0.0168157, 0.0298576, -0.0270253, 0.0726921,
        0.0145232, 0.123291]
        self.A13tevNLO_unc = [2.09078, 10.1517, 0.282307, 0.101205, 1.33191, -8.51168, -1.37309, 2.82636, 1.45767, -4.91761, -0.675197, 1.86189, 0.321422, -0.836276, -0.568156]
        print "initialize"


    # Declare the function
    def functionGF(self, kl,kt,c2,cg,c2g,A): return A[0]*kt**4 + A[1]*c2**2 + (A[2]*kt**2 + A[3]*cg**2)*kl**2  + A[4]*c2g**2 + ( A[5]*c2 + A[6]*kt*kl )*kt**2  + (A[7]*kt*kl + A[8]*cg*kl )*c2 + A[9]*c2*c2g  + (A[10]*cg*kl + A[11]*c2g)*kt**2+ (A[12]*kl*cg + A[13]*c2g )*kt*kl + A[14]*cg*c2g*kl

    def functionGFNLO(self, kl,kt,c2,cg,c2g,A):
        return A[0]*kt**4 + A[1]*c2**2 + (A[2]*kt**2 + A[3]*cg**2)*kl**2  +\
               A[4]*c2g**2 + ( A[5]*c2 + A[6]*kt*kl )*kt**2  + (A[7]*kt*kl + A[8]*cg*kl)*c2 +\
               A[9]*c2*c2g  + (A[10]*cg*kl + A[11]*c2g)*kt**2 +\
               (A[12]*kl*cg + A[13]*c2g)*kt*kl + A[14]*cg*c2g*kl+\
               (A[15]*kt**2 + A[16]*c2)*cg*kt + (A[17]*cg**2*kl + A[18]*c2g*cg)*kt +\
               (A[19]*kt**2 + A[20]*c2**2 + A[21]*cg*kl + A[22]*c2g)*cg**2

    def ReadCoefficientsNLO(self) :
        #for coef in range (0,23) : self.ANLO.append((self.binGenMHHNLO))
        f = open("../data/NLO-Ais.csv", 'r+')
        #countermhh=0
        lines = f.readlines()
        #print len(lines)
        for ll, line in enumerate(lines):
          if ll > len(lines) - 3 : continue
          tokens = line.split(",")
          #if ll==0 : print (len(tokens), tokens)
          for tt, token in enumerate(tokens) :
              if tt == 0 : continue
              self.ANLO[tt-1][ll] = float(token)
          #countermhh = countermhh + 1
        print (len(self.ANLO), len(self.ANLO[0]), self.ANLO[0])

    def ReadCoefficients2(self, energy, cms_base) :
        self.fileCoef13TeV = ROOT.TFile(cms_base+"/src/HHStatAnalysis/AnalyticalModels/data/Coefficients_13TeV.root")
        self.fileCoef14TeV = ROOT.TFile(cms_base+"/src/HHStatAnalysis/AnalyticalModels/data/Coefficients_14TeV.root")
        if energy == 13 :
            self.AbinTot = self.A13tev
            self.fileCoef = self.fileCoef13TeV
        if energy == 14 :
            self.AbinTot = self.A14tev
            self.fileCoef = self.fileCoef14TeV
        for coef in xrange(1,16) :
            histo = "A"+str(coef)+"_"+str(int(energy))+"TeV"
            dumb = self.fileCoef.Get(histo)
            self.AbinHist.append(dumb)
            print ("append histos", histo, dumb.Integral())
        print len(self.AbinTot)

    def getScaleFactor2(self, mhh, cost, kl, kt, c2, cg, c2g, inputSamplesHisto) :
       binmhh = self.AbinHist[0].GetXaxis().FindBin(mhh)
       bincost = self.AbinHist[0].GetYaxis().FindBin(cost)
       effSumAll = inputSamplesHisto.GetBinContent(binmhh, bincost)
       if effSumAll > 0 :
          Abin=[]
          for coef in range (0,15) :
              Abin.append(self.AbinTot[coef]*self.AbinHist[coef].GetBinContent(binmhh, bincost))
          CalcWeight = self.functionGF(kl, kt, c2, cg, c2g, Abin)/effSumAll
          return CalcWeight
       else :
           print (mhh, cost, "return 0", effSumAll)
           return 0

    def getScaleFactor_fromSMonly(self, mhh, cost, kl, kt, c2, cg, c2g) :
       binmhh = self.AbinHist[0].GetXaxis().FindBin(mhh)
       bincost = self.AbinHist[0].GetYaxis().FindBin(cost)
       Abin=[]
       for coef in range (0,15) :
          Abin.append(self.AbinTot[coef]*self.AbinHist[coef].GetBinContent(binmhh, bincost))
       CalcWeight = self.functionGF(kl, kt, c2, cg, c2g, Abin)
       return CalcWeight

    def getNormalization2(self,kl, kt,c2,cg,c2g):
      sumOfWeights = 0
      totCX = self.functionGF(kl,kt,c2,cg,c2g,self.AbinTot)
      if self.functionGF(kl,kt,c2,cg,c2g,self.AbinTot) ==0 : print "total Cross Section = 0 , chose another point"
      for binmhh in range (1, self.AbinHist[0].GetNbinsX()+1) :
         for bincost in range (1, self.AbinHist[0].GetNbinsY()+1) :
            Abin=[]
            for coef in range(0,15) :
                Abin.append( self.AbinTot[coef]*self.AbinHist[coef].GetBinContent(binmhh, bincost) )
            sumOfWeights += self.functionGF(kl,kt,c2,cg,c2g,Abin)/totCX
      print ("calcSumOfWeights", sumOfWeights/self.neventHist)
      return (totCX*sumOfWeights) ## it takes into account the rest of the normalizations

    def getInputInHisto(self, inputSamples, inputSamplestree) :
        fileAll = ROOT.TFile(inputSamples)
        tree = fileAll.Get(inputSamplestree)
        histTotal = self.AbinHist[0].Clone()
        histTotal.Reset()
        print tree.GetEntries()
        for event in tree : histTotal.Fill(event.Genmhh, abs(event.GenHHCost))
        print histTotal.Integral()
        histTotal.SetName("EventsSum")
        fileAllSave = ROOT.TFile("HistoInputEvents.root","RECREATE")
        fileAllSave.WriteTObject(histTotal, "EventsSum", 'Overwrite')
        fileAllSave.Close()

    # taking coefficients from reference XXXX
    def getScaleFactorNLO(self,mhh , cost,kl, kt,c2,cg,c2g, effSumV0,Cnorm) :
       binmhh = 0
       bincost = 0
       if mhh < 247. : print "These variables are problably not genLevel (mhh < 250)"
       for ii in range (0,len(self.binGenMHHNLO)) :
         if mhh >= self.binGenMHHNLO[len(self.binGenMHHNLO)-1-ii] :
            binmhh = self.NMHHbin-1-ii
            break
       if effSumV0 > 0 :
          Abin=[]
          for coef in range (0,23) : Abin.append(self.ANLO[coef][binmhh])
          print ("Abin" , Abin) #### I need an NLO sample ?
          effSMNLO = self.functionGFNLO(1.0, 1.0, 0.0, 0.0, 0.0, Abin)
          print ("effSMNLO", effSMNLO)
          effBSM = float((effSMNLO)*self.functionGFNLO(kl,kt,c2,cg,c2g,Abin)/self.functionGFNLO(kl,kt,c2,cg,c2g, self.A13tevNLO))
          CalcWeight = ((effBSM)/float(effSumV0))/Cnorm # ==> V0 sum in denominator (Moriond 2016)
          return CalcWeight
       else : return 0

    def ReadCoefficients(self,inputFileName) :
        # here you should return TH2D histogram with BSM/SM coefficientes to calculate the scale factors for m_hh vs. cos_theta_star
        # loop over events and efficency calculation will be channel-dependent, so corresponding code
        # should go to the other file
        for coef in range (0,15) : self.A.append((self.NCostHHbin,self.NMHHbin))
        #for coef in range (0,15) : self.A.append((self.NCostHHbin,self.NMHHbin))
        f = open(inputFileName, 'r+')
        lines = f.readlines() # get all lines as a list (array)
        # Read coefficients by bin
        countercost=0
        countermhh=0
        for line in  lines:
          l = []
          tokens = line.split()
          for token in tokens:
              num = ""
              num_char = "."
              num2 = "e"
              num3 = "-"
              for char in token:
                  if (char.isdigit() or (char in num_char) or (char in num2) or (char in num3)): num = num + char
              try: l.append(float(num))
              except ValueError: pass
          self.MHH[countercost][countermhh] = l[1]
          self.COSTS[countercost][countermhh] = l[2]
          self.effSM[countercost][countermhh] = l[3] /100000. # in units of 10k events
          self.effSum[countercost][countermhh] = l[4] /100000. # in units of 10k events # 12 JHEP benchmarks
          # Just for testing purposes the above contains the number of events by bin from an ensenble of events
          # calculated from the 12 benchmarks defined in 1507.02245v4 (JHEP version) each one with 100k events
          for coef in range (0,15) : self.A[coef][countercost][countermhh] = l[5+coef]
          countercost+=1
          if countercost == self.NCostHHbin : # 5
             countercost=0
             countermhh+=1
        f.close()

        # and at the end of the function return it
        print "Stored coefficients by bin"

    def getNormalization(self,kl, kt,c2,cg,c2g,HistoAllEventsName,histfiletitle):
      fileHH=ROOT.TFile(HistoAllEventsName)
      HistoAllEvents = fileHH.Get(histfiletitle)
      sumOfWeights = 0
      #print ("Nbins", HistoAllEvents.GetNbinsX(),HistoAllEvents.GetNbinsY())
      totCX = self.functionGF(kl,kt,c2,cg,c2g,self.A13tev)
      if self.functionGF(kl,kt,c2,cg,c2g,self.A13tev) ==0 : print "total Cross Section = 0 , chose another point"
      for binmhh in range (0,HistoAllEvents.GetNbinsX()) :
         for bincost in range (0,HistoAllEvents.GetNbinsY()) :
            Abin=[]
            for coef in range (0,15) : Abin.append(self.A[coef][bincost][binmhh] )
            sumOfWeights+=float((self.effSM[bincost][binmhh]/30.0)*self.functionGF(kl,kt,c2,cg,c2g,Abin)/totCX)
      fileHH.Close()
      #self.Cnorm = float(sumOfWeights)
      return float(sumOfWeights)

    # distribute the calculated GenMHH and CostS in the bins numbering  (matching the coefficientsByBin_klkt.txt)
    def getScaleFactor(self, mhh, cost,kl, kt,c2,cg,c2g, effSumV0, Cnorm) :
       binmhh = 0
       bincost = 0
       if mhh < 247. : print "These variables are problably not genLevel (mhh < 250)"
       for ii in range (0,self.NMHHbin) :
         if mhh >= self.binGenMHH[self.NMHHbin-1-ii] :
            binmhh = self.NMHHbin-1-ii
            break
       for ii in range (0,self.NCostHHbin) :
         var = abs(cost)
         if var >= self.binGenCostS[self.NCostHHbin-1-ii] :
            bincost = self.NCostHHbin-1-ii
            break
       if effSumV0 > 0 :
          Abin=[]
          for coef in range (0,15) : Abin.append(self.A[coef][bincost][binmhh] )
          effBSM = float((self.effSM[bincost][binmhh]/30)*self.functionGF(kl,kt,c2,cg,c2g,Abin)/self.functionGF(kl,kt,c2,cg,c2g,self.A13tev))
          CalcWeight = ((effBSM)/float(effSumV0))/Cnorm # ==> V0 sum in denominator (Moriond 2016)
          return CalcWeight
       else : return 0

    def FindBin(self,mhh,cost,histfilename,histfiletitle) :
       fileHH=ROOT.TFile(histfilename) #Distros_5p_SM3M_sumBenchJHEP_13TeV.root") # do the histo from V0
       histfile = fileHH.Get(histfiletitle)
       bmhh = histfile.GetXaxis().FindBin(mhh)
       #if self.NCostHHbin ==5 or self.NCostHHbin ==4 or self.NCostHHbin ==3 :
       #var = abs(cost)
       var = cost
       #elif self.NCostHHbin ==3 : var = cost
       bcost = histfile.GetYaxis().FindBin(var)
       effSumV0 = histfile.GetBinContent(bmhh,bcost)
       fileHH.Close()
       #print (mhh,cost,bmhh,bcost,effSumV0)
       return effSumV0

    def getCluster(self,kl, kt,c2,cg,c2g,HistoAllEventsName,histfiletitle):
      print "Calculating TS"
      normEv = 1200000000
      # load benchmarks
      self.klJHEP=[1.0,  7.5,  1.0,  1.0,  -3.5, 1.0, 2.4, 5.0, 15.0, 1.0, 10.0, 2.4, 15.0]
      self.ktJHEP=[1.0,  1.0,  1.0,  1.0,  1.5,  1.0, 1.0, 1.0, 1.0,  1.0, 1.5,  1.0, 1.0]
      self.c2JHEP=[0.0,  -1.0, 0.5, -1.5, -3.0,  0.0, 0.0, 0.0, 0.0,  1.0, -1.0, 0.0, 1.0]
      self.cgJHEP=[0.0,  0.0, -0.8,  0.0, 0.0,   0.8, 0.2, 0.2, -1.0, -0.6, 0.0, 1.0, 0.0]
      self.c2gJHEP=[0.0, 0.0, 0.6, -0.8, 0.0, -1.0, -0.2,-0.2,  1.0,  0.6, 0.0, -1.0, 0.0]
      TS = np.zeros(13)
      Cnorm = self.getNormalization(kl, kt,c2,cg,c2g,HistoAllEventsName,histfiletitle)
      for bench in range(0,13) :
        for bincost in range (self.NCostHHbin) :
           for binmhh in range (0, self.NMHHbin) : #48,3 ) : # (merge 3 mhh bins)
              Abin=[]
              for coef in range (0,15) : Abin.append(self.A[coef][bincost][binmhh] )
              mhh = self.binGenMHH[binmhh]+5.0
              cost = self.binGenCostS[bincost]+0.1
              effSumV0 = self.FindBin(mhh,cost,HistoAllEventsName,histfiletitle)
              NevScan = normEv*(float((self.effSM[bincost][binmhh]/30)*self.functionGF(kl,kt,c2,cg,c2g,Abin)/self.functionGF(kl,kt,c2,cg,c2g,self.A13tev))/float(effSumV0))/Cnorm
              klB = self.klJHEP[bench]
              ktB = self.ktJHEP[bench]
              c2B = self.c2JHEP[bench]
              cgB = self.cgJHEP[bench]
              c2gB =self.c2gJHEP[bench]
              CnormB = self.getNormalization(klB, ktB,c2B,cgB,c2gB,HistoAllEventsName,histfiletitle)
              Nbench = normEv*(float((self.effSM[bincost][binmhh]/30)*self.functionGF(klB,ktB,c2B,cgB,c2gB,Abin)/self.functionGF(klB,ktB,c2B,cgB,c2gB,self.A13tev))/float(effSumV0))/CnormB
              NevScanInt =  int(math.floor(NevScan))
              NbenchInt =  int(math.floor(Nbench))
              if NevScanInt <= 0 : NevScanInt =1
              if NbenchInt <= 0 : NbenchInt =1
              NSumInt = (NevScanInt+NbenchInt)/2
              TS[bench]+=-2*(math.log(math.factorial(NevScanInt)) + math.log(math.factorial(NbenchInt)) -2*math.log(math.factorial(NSumInt)) )
      print TS
      minTS = np.argmax(TS)
      return int(minTS)

    ### only to read the text files to test
    def ReadLine(self,line, countline,Px,Py,Pz,En) :
            l = []
            tokens = line.split()
            for token in tokens:
                num = ""
                num_char = "."
                num2 = "e"
                num3 = "-"
                for char in token:
                    if (char.isdigit() or (char in num_char) or (char in num2) or (char in num3)): num = num + char
                try: l.append(float(num))
                except ValueError: pass
            if countline < 2 :
               Px[countline] = l[1]
               Py[countline] = l[2]
               Pz[countline] = l[3]
               En[countline] = l[4]
            #return countline

    def CalculateMhhCost(self,mhhcost,countline,Px,Py,Pz,En) :
               # calculate reweigthing
               if abs(Px[0])!= abs(Px[1]) : print "error parsing ascii file"
               P1 = ROOT.TLorentzVector()
               P1.SetPxPyPzE(Px[0],Py[0],Pz[0],En[0])
               P2 = ROOT.TLorentzVector()
               P1.SetPxPyPzE(Px[1],Py[1],Pz[1],En[1])
               SUM = ROOT.TLorentzVector()
               SUM.SetPxPyPzE(Px[0]+Px[1],Py[0]+Py[1],Pz[0]+Pz[1],En[0]+En[1])
               mhhcost[0]=SUM.M()
               P1boost = P1
               P1boost.Boost(-SUM.BoostVector())
               mhhcost[1] = float(P1boost.CosTheta())
               mhhcost[2] = float(P1.Pt())
               mhhcost[3] = float(SUM.Pt())

    def LoadTestEvents(self,CalcMhhTest,CalcCostTest,filne) :
       counteventSM = 0
       Px = np.zeros((2))
       Py = np.zeros((2))
       Pz = np.zeros((2))
       En = np.zeros((2))
       f = open(filne, 'r+')
       lines = f.readlines() # get all lines as a list (array)
       countline = 0 # particuliarity of the text file with events = each 2 lines are one event there
       for line in  lines:
             self.ReadLine(line, countline,Px,Py,Pz,En)
             #print countline
             countline+=1
             mhhcost= [0,0,0,0] # to store [mhh , cost] of that event
             if countline==2 : # if read 2 lines
                self.CalculateMhhCost(mhhcost,countline,Px,Py,Pz,En) # ==> adapt to your input
                countline=0
                CalcMhhTest[counteventSM] = float(mhhcost[0])
                var = abs(mhhcost[1])
                CalcCostTest[counteventSM] = float(var)
                counteventSM+=1


    ###################################################
    # Draw the histograms
    #####################################################
    def plotting(self,kl,kt,c2,cg,c2g, CalcMhh,CalcCost,CalcWeight,CalcMhhTest,CalcCostTest,drawtest):
      #
      # Set the font dictionaries (for plot title and axis titles)
      matplotlib.rc('xtick', labelsize=16.5)
      matplotlib.rc('ytick', labelsize=16.5)
      title_font = {'size':'16', 'color':'black', 'weight':'normal',
              'verticalalignment':'bottom'} # Bottom vertical alignment for more space
      axis_font = {'size':'16'}
      if drawtest ==0 :
         label = str("SM")
         labeltext =str("$\kappa_{\lambda}$ = "+str(kl)+", $\kappa_{t}$ = "+str(kt)+", $c_{2}$ =  $c_{g}$ =  $c_{2g}$ =" +str(c2g))
      elif drawtest > 0 :
         label =str("BM"+str(drawtest))
         labeltext=str("BM "+str(drawtest))
      else :
         label = str("kl_"+str(kl)+"_kt_"+str(kt)+"_c2_"+str(c2)+"_cg_"+str(cg)+"_c2g_" +str(c2g))
         labeltext =str("kl ="+str(kl)+", kt ="+str(kt)+", c2 ="+str(c2)+", cg ="+str(cg)+", c2g =" +str(c2g))
      #
      print "Plotting test histograms"
      #f, axes = plt.subplots(1,1)
      #f.add_subplot(111, aspect='equal')
      bin_size = 50; min_edge = 200; max_edge = 1500
      N = (max_edge-min_edge)/bin_size; Nplus1 = N + 1
      bin_list = np.linspace(min_edge, max_edge, Nplus1)
      plt.subplots(figsize=(6, 6))
      plt.xlim(min_edge, max_edge)
      #plt.ylim(0.0, 0.01)
      plt.hist(CalcMhh, bin_list, weights=CalcWeight, histtype='bar', label='reweigted', fill=False, color= 'g', edgecolor='g', lw=5 ,alpha = 1 , normed=1)
      if drawtest>-2 :
         n, bins, _ = plt.hist(CalcMhhTest, bins=bin_list,  histtype='bar', label='simulated', fill=False, color= 'r', edgecolor='r', lw=1 ,alpha = 1, normed=1) #
         mid = 0.5*(bins[1:] + bins[:-1])
         plt.errorbar(mid, n/float(len(CalcWeight)), yerr=np.sqrt(n)/float(len(CalcWeight)), fmt='none', color= 'r', ecolor= 'r', edgecolor='r', normed=1)
      plt.legend(loc='upper right',title=labeltext)
      plt.subplots_adjust(left=0.15,bottom=0.15)
      plt.xlabel("$M_{HH}^{Gen}$ (GeV)", **axis_font)
      plt.ylabel("a.u.", **axis_font)
      plt.savefig("MhhGen_"+label+".pdf")
      plt.savefig("MhhGen_"+label+".png")
      plt.cla()   # Clear axis
      plt.clf()   # Clear figure
      plt.close()
      #
      bin_size = 0.05; min_edge = 0; max_edge = 1
      N = (max_edge-min_edge)/bin_size; Nplus1 = N + 1
      bin_list = np.linspace(min_edge, max_edge, Nplus1)
      plt.subplots(figsize=(6, 6))
      plt.xlim(min_edge, max_edge)
      plt.hist(CalcCost, bin_list, weights=CalcWeight , histtype='bar', label='reweigted', fill=False, color= 'g', edgecolor='g', lw=5 ,alpha = 1 , normed=1)
      if drawtest>-2 :
         n, bins, _ = plt.hist(CalcCostTest, bin_list ,  histtype='bar', label='simulated',fill=False, color= 'r', edgecolor='r', lw=1 ,alpha = 1 , normed=1)
         mid = 0.5*(bins[1:] + bins[:-1])
         err=np.sqrt(n)
         plt.errorbar(mid, n, yerr=err, fmt='none', color= 'r', ecolor= 'r', edgecolor='r')
      plt.legend(loc='upper right',title=labeltext,fontsize = 'large')
      plt.xlabel("$cost\theta^*HH^{Gen}$")
      plt.ylabel("a.u.")
      plt.ylim(0, 350)
      plt.subplots_adjust(left=0.1)
      plt.savefig("CostS_"+label+".pdf")
      plt.savefig("CostS_"+label+".png")
      plt.cla()   # Clear axis
      plt.clf()   # Clear figure
      plt.close()
