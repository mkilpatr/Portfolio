import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True
import math
import numpy as np
from functools import reduce
import operator
import os

from PhysicsTools.NanoAODTools.postprocessing.framework.datamodel import Collection, Object
from PhysicsTools.NanoAODTools.postprocessing.framework.eventloop import Module
from PhysicsTools.NanoSUSYTools.modules.datamodelRemap import ObjectRemapped, CollectionRemapped
from PhysicsTools.NanoAODTools.postprocessing.tools import deltaPhi, deltaR, closest
from PhysicsTools.NanoSUSYTools.modules.Stop0lObjectsProducer import DeepCSVMediumWP, DeepCSVLooseWP

class LLObjectsProducer(Module):
    def __init__(self, era, Process, isData = False, applyUncert=None):
        self.era = era
        self.process = Process
        self.isData = isData
        self.metBranchName = "MET"
        self.applyUncert = applyUncert
        self.suffix = ""

        if self.applyUncert == "JESUp":
            self.suffix = "_JESUp"
        elif self.applyUncert == "METUnClustUp":
            self.suffix = "_METUnClustUp"
        elif self.applyUncert == "JESDown":
            self.suffix = "_JESDown"
        elif self.applyUncert == "METUnClustDown":
            self.suffix = "_METUnClustDown"

    def beginJob(self):
        pass
    def endJob(self):
        pass

    def loadhisto(self,filename,hname):
        file =ROOT.TFile.Open(filename)
        hist_ = file.Get(hname)
        hist_.SetDirectory(0)
        return hist_

    def beginFile(self, inputFile, outputFile, inputTree, wrappedOutputTree):
        self.isFirstEventOfFile = True
        self.out = wrappedOutputTree
        self.out.branch("Stop0l_MtLepMET"		+ self.suffix, 	"F")
        self.out.branch("Stop0l_nVetoElecMuon"		+ self.suffix, 	"I")
        self.out.branch("Stop0l_nVetoElectron"	        + self.suffix, 	"I")
        self.out.branch("Stop0l_nVetoMuon"  	        + self.suffix, 	"I")
        self.out.branch("Stop0l_noMuonJet"		+ self.suffix,	"O")
        self.out.branch("Pass_dPhiQCD"			+ self.suffix,	"O")
        self.out.branch("Pass_dPhiQCDSF"		+ self.suffix,	"O")
        self.out.branch("Stop0l_dPhiISRMET"		+ self.suffix,	"F")
        self.out.branch("Pass_exHEMVetoElec30"		+ self.suffix,  "O")
        self.out.branch("Pass_exHEMVetoPho30"		+ self.suffix,  "O")
        self.out.branch("Pass_exHEMVetoJet30"		+ self.suffix,  "O")
        self.out.branch("Pass_LHETTbar"			+ self.suffix,  "O")
        self.out.branch("LHEScaleWeight_Up"		+ self.suffix,  "F")
        self.out.branch("LHEScaleWeight_Down"		+ self.suffix,  "F")
        self.out.branch("genMatchedLep"			+ self.suffix,  "I")
        if not self.isData:
            self.out.branch("ElectronVetoCRSF"		+ self.suffix,	"F")
            self.out.branch("ElectronVetoCRSFErr"	+ self.suffix,  "F")
            self.out.branch("ElectronVetoSRSF"		+ self.suffix,	"F")
            self.out.branch("ElectronVetoSRSFErr"	+ self.suffix,  "F")
            self.out.branch("MuonLooseCRSF"		+ self.suffix,	"F")
            self.out.branch("MuonLooseCRSFErr"		+ self.suffix,	"F")
            self.out.branch("MuonLooseSRSF"		+ self.suffix,	"F")
            self.out.branch("MuonLooseSRSFErr"		+ self.suffix,	"F")
            self.out.branch("TauCRSF"			+ self.suffix,	"F")
            self.out.branch("TauCRSF_Up"		+ self.suffix,	"F")
            self.out.branch("TauCRSF_Down"		+ self.suffix,	"F")
            self.out.branch("TauSRSF"			+ self.suffix,	"F")
            self.out.branch("TauSRSF_Up"		+ self.suffix,	"F")
            self.out.branch("TauSRSF_Down"		+ self.suffix,	"F")
            self.out.branch("SoftBSF"			+ self.suffix,	"F")
            self.out.branch("SoftBSFErr"		+ self.suffix,	"F")

    def GetJetSortedIdx(self, jets, jetpt = 20, jeteta = 4.7):
        ptlist = []
        etalist = []
        dphiMET = []
        for j in jets:
            if math.fabs(j.eta) > jeteta or j.pt < jetpt:
                pass
            else:
                ptlist.append(-j.pt)
                etalist.append(math.fabs(j.eta))
                dphiMET.append(j.dPhiMET)
        
        sortIdx = np.lexsort((etalist, ptlist))
        
        return sortIdx, [dphiMET[j] for j in sortIdx]

    def PassdPhi(self, sortedPhi, dPhiCuts, invertdPhi =False):
        if invertdPhi:
            return any( a < b for a, b in zip(sortedPhi, dPhiCuts))
        else:
            return all( a > b for a, b in zip(sortedPhi, dPhiCuts))

    def PassdPhiVal(self, sortedPhi, dPhiCutsLow, dPhiCutsHigh):
        return all( (a < b and b < c) for a, b, c in zip(dPhiCutsLow, sortedPhi, dPhiCutsHigh))

    def SelNoMuon(self, jets, met):
        noMuonJet = True
        for j in jets:
            if j.pt > 200 and j.muEF > 0.5 and abs(deltaPhi(j.phi, met.phi)) > (math.pi - 0.4):
                noMuonJet = False
        return noMuonJet

    def isA(self, particleID, p):
        return abs(p) == particleID


    def ScaleFactorErrElectron(self, obj, kind="Veto", region="CR"):
        sf = 1
        sfErr = 0
        pt_comp = 99999.
        for s in obj:
            if not s.Stop0l: continue
            if region == "CR":
                if kind == "Medium":
                    sf *= s.MediumSF
                    sfErr += ((s.MediumSFErr)**2)
                elif kind == "Veto":
                    sf *= s.VetoSF
                    sfErr += ((s.VetoSFErr)**2)
            if s.pt < pt_comp and region == "SR":
                pt_comp = s.pt
                sf = s.VetoSF
                sfErr = ((s.VetoSFErr)**2)
        
        return sf, math.sqrt(sfErr)

    def ScaleFactorErrMuon(self, obj, kind="Loose", region="CR"):
        sf = 1
        sfErr = 0
        pt_comp = 99999.
        for s in obj:
            if not s.Stop0l: continue
            if region == "CR":
                if kind == "Loose":
                    sf *= s.LooseSF
                    sfErr += ((s.LooseSFErr)**2)
                elif kind == "Medium":
                    sf *= s.MediumSF
                    sfErr += ((s.MediumSFErr)**2)
            if s.pt < pt_comp and region == "SR":
                sf = s.LooseSF
                sfErr = ((s.LooseSFErr)**2)
        
        return sf, math.sqrt(sfErr)

    def ScaleFactorErrTau(self, obj, region = "CR"):
        sf = 1
        sfUp = 0
        sfDown = 0
        pt_comp = 99999.
        for s in obj:
            if not s.Stop0l: continue
            if region == "CR":
                sf *= s.MediumSF
                sfUp += ((s.MediumSF_Up)**2)
                sfDown += ((s.MediumSF_Down)**2)
            if s.pt < pt_comp and region == "SR":
                sf = s.MediumSF
                sfUp = ((s.MediumSF_Up)**2)
                sfDown = ((s.MediumSF_Down)**2)
        
        return sf, math.sqrt(sfUp), math.sqrt(sfDown)

    def ScaleFactorErrSoftB(self, obj):
        sf = 1
        sfErr = 0
        for s in obj:
            if not s.Stop0l: continue
            sf *= s.SF
            sfErr += ((s.SFerr)**2)
        
        return sf, math.sqrt(sfErr)

    def PassObjectVeto(self, lep, eta_low, eta_high, phi_low, phi_high):
        for l in lep:
            if not l.Stop0l: continue
            if l.eta >= eta_low and l.eta <= eta_high and l.phi >= phi_low and l.phi <= phi_high:
                return False
        return True

    def HEMVetoLepton(self, ele, pho, jet):
        narrow_eta_low  = -3.0
        narrow_eta_high = -1.4
        narrow_phi_low  = -1.57
        narrow_phi_high = -0.87
        wide_eta_low    = -3.2
        wide_eta_high   = -1.2
        wide_phi_low    = -1.77
        wide_phi_high   = -0.67
        
        Pass_HEMveto_ele = self.PassObjectVeto(ele, narrow_eta_low, narrow_eta_high, narrow_phi_low, narrow_phi_high)
        Pass_HEMveto_pho = self.PassObjectVeto(pho, narrow_eta_low, narrow_eta_high, narrow_phi_low, narrow_phi_high)
        Pass_HEMveto_jet = self.PassObjectVeto(jet, wide_eta_low, wide_eta_high, wide_phi_low, wide_phi_high)
        return Pass_HEMveto_ele, Pass_HEMveto_pho, Pass_HEMveto_jet

    def LHEScale(self, lhewgt):
        LHEwgt_Up = -1.
        LHEwgt_Down = 10000.

	try:
            LHEwgt_Up = 1.
            LHEwgt_Down = 1.
            for l in xrange(len(lhewgt)):
                    temp_min = lhewgt[l]
                    temp_max = lhewgt[l]
                    LHEwgt_Up = max(temp_min, LHEwgt_Up)
                    LHEwgt_Down = min(temp_max, LHEwgt_Down)
        except SystemError:
            return 1.0, 1.0

        if abs(LHEwgt_Up) > 10000000000: LHEwgt_Up = 1.
        if abs(LHEwgt_Down) > 10000000000: LHEwgt_Down = 1.

        return LHEwgt_Up, LHEwgt_Down       

    def genLepMatch(self, genpars, el, mu):
        #for genpar in genpars :     #loop on genpars       
        #print len(genpars)
        isChargedLep = False
        for i in range(len(genpars)):
            genpar = genpars[i]
            if abs(genpar.pdgId)==11 or abs(genpar.pdgId)==13 or abs(genpar.pdgId)==15:
                if (((genpar.statusFlags & 0x2100) == 0x2100) or ((genpar.statusFlags & 0x2080) == 0x2080)): 
                    isChargedLep = True
                    break

        return isChargedLep

    def getattr_safe(self, event, name):
        out = None
        try:
            out = getattr(event, name)
        except RuntimeError:
            pass
        return out

    def analyze(self, event):
        """process event, return True (go to next module) or False (fail, go to next event)"""
        ## Getting objects
        if not self.isData: genpart   = Collection(event, "GenPart")
        electrons = Collection(event, "Electron")
        photons   = Collection(event, "Photon")
        muons     = Collection(event, "Muon")
        isotracks = Collection(event, "IsoTrack")
        taus      = Collection(event, "Tau")
        stop0l    = Object(event,     "Stop0l")
        fatjets   = Collection(event, "FatJet")
        SB        = Collection(event, "SB")
        restop    = Collection(event, "ResolvedTopCandidate")
        res       = Collection(event, "ResolvedTop", lenVar="nResolvedTopCandidate")
        jets      = Collection(event, "Jet")
        met       = Object(event, self.metBranchName)
        lhe       = Object(event, "LHE")
        if "TTbar" in self.process and self.era == "2016":
            lhewgt = event.LHEScaleWeight
            lhevec = [lhewgt[0], lhewgt[1], lhewgt[3], lhewgt[4], lhewgt[5], lhewgt[7], lhewgt[8]]

        if self.applyUncert == "JESUp":
            jets      = CollectionRemapped(event, "Jet", replaceMap={"pt":"pt_jesTotalUp", "mass":"mass_jesTotalUp"})
            met       = ObjectRemapped(event, self.metBranchName, replaceMap={"pt":"pt_jesTotalUp", "phi":"phi_jesTotalUp"})
        elif self.applyUncert == "JESDown":
            jets      = CollectionRemapped(event, "Jet", replaceMap={"pt":"pt_jesTotalDown", "mass":"mass_jesTotalDown"})
            met       = ObjectRemapped(event, self.metBranchName, replaceMap={"pt":"pt_jesTotalDown", "phi":"phi_jesTotalDown"})
        elif self.applyUncert == "METUnClustUp":
            jets      = Collection(event, "Jet")
            met       = ObjectRemapped(event, self.metBranchName, replaceMap={"pt":"pt_unclustEnUp", "phi":"phi_unclustEnUp"})
        elif self.applyUncert == "METUnClustDown":
            jets      = Collection(event, "Jet")
            met       = ObjectRemapped(event, self.metBranchName, replaceMap={"pt":"pt_unclustEnDown", "phi":"phi_unclustEnDown"})
        
        ## Selecting objects
        mt                   = sum([ e.MtW for e in electrons if e.Stop0l ] + [ m.MtW for m in muons if m.Stop0l ])
        countEle             = sum([e.Stop0l for e in electrons])
        countMuon            = sum([m.Stop0l for m in muons])
        noMuonJet            = self.SelNoMuon(jets, met)
        sortedIdx, sortedPhi = self.GetJetSortedIdx(jets)
        PassdPhiQCD          = self.PassdPhi(sortedPhi, [0.1, 0.1, 0.1], invertdPhi =True)
        PassdPhiQCDSF        = self.PassdPhi(sortedPhi, [0.1, 0.1], invertdPhi =True)
        dphiISRMet           = abs(deltaPhi(fatjets[stop0l.ISRJetIdx].phi, met.phi)) if stop0l.ISRJetIdx >= 0 else -1
        Pass_HEMElec, Pass_HEMPho, Pass_HEMJet = self.HEMVetoLepton(electrons, photons, jets)
        PassLHE              = lhe.HTIncoming < 600 if (("DiLep" in self.process) or ("SingleLep" in self.process)) else True 

        if not self.isData:
            if "TTbar" in self.process and self.era == "2016":
                LHEwgt_Up, LHEwgt_Down 			= self.LHEScale(lhevec)
            else:
                LHEwgt_Up = 1.0
                LHEwgt_Down = 1.0
            electronVetoCRSF, electronVetoCRSFErr       = self.ScaleFactorErrElectron(electrons, "Veto", "CR")
            electronVetoSRSF, electronVetoSRSFErr       = self.ScaleFactorErrElectron(electrons, "Veto", "SR")
            muonLooseCRSF, muonLooseCRSFErr             = self.ScaleFactorErrMuon(muons, "Loose", "CR")
            muonLooseSRSF, muonLooseSRSFErr             = self.ScaleFactorErrMuon(muons, "Loose", "SR")
            tauCRSF, tauCRSFUp, tauCRSFDown             = self.ScaleFactorErrTau(taus, "CR")
            tauSRSF, tauSRSFUp, tauSRSFDown             = self.ScaleFactorErrTau(taus, "SR")
            ## type top = 1, W = 2, else 0
            softBSF, softBSFErr                         = self.ScaleFactorErrSoftB(SB)
            isGenLep					= self.genLepMatch(genpart, electrons, muons)

        ### Store output
        self.out.fillBranch("Stop0l_MtLepMET"		+ self.suffix,  mt)
        self.out.fillBranch("Stop0l_nVetoElecMuon"	+ self.suffix, 	countEle + countMuon)
        self.out.fillBranch("Stop0l_nVetoElectron"	+ self.suffix, 	countEle)
        self.out.fillBranch("Stop0l_nVetoMuon"  	+ self.suffix, 	countMuon)
        self.out.fillBranch("Stop0l_noMuonJet"		+ self.suffix,	noMuonJet)
        self.out.fillBranch("Pass_dPhiQCD"		+ self.suffix,	PassdPhiQCD)
        self.out.fillBranch("Pass_dPhiQCDSF"		+ self.suffix,	PassdPhiQCDSF)
        self.out.fillBranch("Stop0l_dPhiISRMET"		+ self.suffix,	dphiISRMet)
        self.out.fillBranch("Pass_exHEMVetoElec30"	+ self.suffix,  Pass_HEMElec)
        self.out.fillBranch("Pass_exHEMVetoPho30"	+ self.suffix,  Pass_HEMPho)
        self.out.fillBranch("Pass_exHEMVetoJet30"	+ self.suffix,  Pass_HEMJet)
        self.out.fillBranch("Pass_LHETTbar"		+ self.suffix,  PassLHE)
        
        if not self.isData:
            self.out.fillBranch("genMatchedLep"		+ self.suffix,  isGenLep)
            self.out.fillBranch("LHEScaleWeight_Up"	+ self.suffix,  LHEwgt_Up)
            self.out.fillBranch("LHEScaleWeight_Down"	+ self.suffix,  LHEwgt_Down)
            self.out.fillBranch("ElectronVetoCRSF"	+ self.suffix,	electronVetoCRSF)
            self.out.fillBranch("ElectronVetoCRSFErr"	+ self.suffix,  electronVetoCRSFErr)
            self.out.fillBranch("ElectronVetoSRSF"	+ self.suffix,	electronVetoSRSF)
            self.out.fillBranch("ElectronVetoSRSFErr"	+ self.suffix,  electronVetoSRSFErr)
            self.out.fillBranch("MuonLooseCRSF"		+ self.suffix,	muonLooseCRSF)
            self.out.fillBranch("MuonLooseCRSFErr"	+ self.suffix,	muonLooseCRSFErr)
            self.out.fillBranch("MuonLooseSRSF"		+ self.suffix,	muonLooseSRSF)
            self.out.fillBranch("MuonLooseSRSFErr"	+ self.suffix,	muonLooseSRSFErr)
            self.out.fillBranch("TauCRSF"		+ self.suffix,	tauCRSF)
            self.out.fillBranch("TauCRSF_Up"		+ self.suffix,	tauCRSFUp)
            self.out.fillBranch("TauCRSF_Down"		+ self.suffix,	tauCRSFDown)
            self.out.fillBranch("TauSRSF"		+ self.suffix,	tauSRSF)
            self.out.fillBranch("TauSRSF_Up"		+ self.suffix,	tauSRSFUp)
            self.out.fillBranch("TauSRSF_Down"		+ self.suffix,	tauSRSFDown)
            self.out.fillBranch("SoftBSF"		+ self.suffix,	softBSF)
            self.out.fillBranch("SoftBSFErr"		+ self.suffix,	softBSFErr)
        return True


 # define modules using the syntax 'name = lambda : constructor' to avoid having them loaded when not needed
