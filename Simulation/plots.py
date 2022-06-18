import numpy as np
#np.set_printoptions(threshold=sys.maxsize)
import ROOT
import math

import argparse
parser = argparse.ArgumentParser(description="run on .npz file with two numpy arrays: gen and gaus")
parser.add_argument("file", help=".npz file path")
parser.add_argument("-l", "--label", help="the label that was sampled, if provided will zoom the plots on this point")
parser.add_argument("-w", "--width", default=0.02, help="the width of the gaussian")
parser.add_argument("-s", "--save", metavar='FILE', help="save to file")
args = parser.parse_args()

npz = np.load(args.file)
gen = npz['gen']
gaus = npz['gaus']

print(gen)
print(gen.shape)
print(gaus)
print(gaus.shape)

hist_gen = ROOT.TH2D('gen', 'Generator Network', 1000, -2, 2, 1000, -2, 2)
hist_gaus = ROOT.TH2D('gaus', '2D Gaussian', 1000, -2, 2, 1000, -2, 2)

for row in range(gen.shape[0]):
  sample = gen[row, :]
  x = sample[0]
  y = sample[1]
  hist_gen.Fill(x, y)
for row in range(gaus.shape[0]):
  sample = gaus[row, :]
  x = sample[0]
  y = sample[1]
  hist_gaus.Fill(x, y)

ROOT.gStyle.SetOptStat(110011)
ROOT.gStyle.SetOptFit(111)

c = ROOT.TCanvas('c', 'c', 1440, 900)
c.Divide(3,2)
c.cd(1)
hist_gaus.Draw('colz')
c.cd(4)
hist_gen.Draw('colz')
c.cd(2)
ROOT.gPad.SetLogy()
hist_gaus_x = hist_gaus.ProjectionX()
hist_gaus_x.SetTitle('X Projection')
hist_gaus_x.Draw('')
hist_gaus_x.Fit('gaus')
c.cd(3)
ROOT.gPad.SetLogy()
hist_gaus_y = hist_gaus.ProjectionY()
hist_gaus_y.SetTitle('Y Projection')
hist_gaus_y.Draw('')
hist_gaus_y.Fit('gaus')
c.cd(5)
ROOT.gPad.SetLogy()
hist_gen_x = hist_gen.ProjectionX()
hist_gen_x.SetTitle('X Projection')
hist_gen_x.Draw('')
hist_gen_x.Fit('gaus')
c.cd(6)
ROOT.gPad.SetLogy()
hist_gen_y = hist_gen.ProjectionY()
hist_gen_y.SetTitle('Y Projection')
hist_gen_y.Draw('')
hist_gen_y.Fit('gaus')

if args.label != None:
  label = float(args.label)
  angle = np.pi/2 - label
  x0 = math.cos(angle)
  y0 = math.sin(angle)
  w = args.width
  xmin = x0 - 10*w
  xmax = x0 + 10*w
  ymin = y0 - 10*w
  ymax = y0 + 10*w
  xaxis = hist_gaus.GetXaxis()
  yaxis = hist_gaus.GetYaxis()
  xaxis.SetRangeUser(xmin,xmax)
  yaxis.SetRangeUser(ymin,ymax)
  xaxis = hist_gen.GetXaxis()
  yaxis = hist_gen.GetYaxis()
  xaxis.SetRangeUser(xmin,xmax)
  yaxis.SetRangeUser(ymin,ymax)
  axis = hist_gaus_x.GetXaxis()
  axis.SetRangeUser(xmin,xmax)
  axis = hist_gaus_y.GetXaxis()
  axis.SetRangeUser(ymin,ymax)
  axis = hist_gen_x.GetXaxis()
  axis.SetRangeUser(xmin,xmax)
  axis = hist_gen_y.GetXaxis()
  axis.SetRangeUser(ymin,ymax)
  c.Modified()

c.Modified()

if not args.save:
  input()
else:
  c.SaveAs(args.save)
