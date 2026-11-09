#import libraries for part 1;
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
#part 2;
from scipy.interpolate import interp1d
#part 3;
import pandas as pd

print("\nName: Luke O Brien\nStudent Number: 23373026")

fitsFile = fits.open('100pc_rvnotnull.fits')
#fitsFile.info()

data = fitsFile[1].data

#umber of sources
numSources = len(data)

#-----
#Part 1
print("\n---\nPart 1")
#task 1
print("\nNumber of sources in the sample:", numSources)
#print(data.columns) #just to visualise


#plt.errorbar(x, y, kwargs) (radial_velocity,)

#extracting the needed columns
parallax = data['PARALLAX_1']
parallaxError = data['PARALLAX_ERROR_1']

bp = data['PHOT_BP_MEAN_MAG_1']
bpFluxoverError = data['PHOT_BP_MEAN_FLUX_OVER_ERROR_1'] #not sure if this is the coressponding error
bpError = (2.5/ (np.log(10) * bpFluxoverError))

rp = data['PHOT_RP_MEAN_MAG_1']
rpFluxoverError = data['PHOT_RP_MEAN_FLUX_OVER_ERROR_1'] #not sure if this is the coressponding error
rpError = (2.5 / (np.log(10) * rpFluxoverError))

g = data['PHOT_G_MEAN_MAG_1']
gError = (2.5 / (np.log(10) * (data['PHOT_G_MEAN_FLUX_OVER_ERROR_1']))) #not sure if this is the coressponding error

radVel = data['radial_velocity']
radVelError = data['radial_velocity_error']
properMotionRA = data['PMRA_1'] #proper motion RA
properMotionRA_Error = data['PMRA_ERROR_1']
properMotionDec = data['PMDEC_1'] #proper motion declination
properMotionDec_Error = data['PMDEC_ERROR_1']



#these are used in part 1 task 8
RA = data['RA_1']
RA_Error = ['RA_ERROR_1']
dec = data['DEC_1']
decError = data['DEC_ERROR_1']

#-----
#task 2
#calculate colour
xAxis = bp - rp

#convert parallax to distance
d = 1000/parallax

#absolute magnitude
M_G = g - 5*np.log10(d) + 5

#plot
plt.scatter(xAxis, M_G, s=1,color = 'black')
plt.gca().invert_yaxis()
plt.xlabel("BP - RP")
plt.ylabel("Absolute Magnitude $M_G$")
plt.title("H-R Diagram")
plt.show()

#-----------

#task 3
 
c1 = 0.97779e9 #1^3 pc km^-1 yr_1
c2 = 4.74047 #AU km^1 yr^1
 
 
N = 1000
 
numSources = len(parallax)
 
#for computing the mean and std, i summed the variables so i dont need to 
#storing the full arrays of 1000 points
tphSum = np.zeros(numSources)
tphSquaredSum = np.zeros(numSources)
dphSum = np.zeros(numSources)
dphSquaredSum = np.zeros(numSources)
vTSum = np.zeros(numSources)
v_ph_sum = np.zeros(numSources)
 
#this saves the informations at these steps for plotting
checkIndex = [0, 300, 1000]
parallaxCheck = {i: np.empty(N) for i in checkIndex}
radVelCheck = {i: np.empty(N) for i in checkIndex}
tphCheck = {i: np.empty(N) for i in checkIndex}
dphCheck = {i: np.empty(N) for i in checkIndex}
 
for i in range(N):
    #random samples from Gaussian distributions
    parallaxSample = np.random.normal(parallax, parallaxError)
    radVelSample = np.random.normal(radVel, radVelError)
    sample_pmRAample = np.random.normal(properMotionRA, properMotionRA_Error)
    pmDecSample = np.random.normal(properMotionDec, properMotionDec_Error)
 
    #apply the same equations to each sample
    transverseVelSample = c2 * np.sqrt(sample_pmRAample**2 + pmDecSample**2) / parallaxSample
    vSample = np.sqrt(transverseVelSample**2 + radVelSample**2)
    tphSample = -c1 * (1/parallaxSample) * (radVelSample / vSample**2)
    dphSample = (1000/parallaxSample) * (transverseVelSample / vSample)
 
    tphSum += tphSample
    tphSquaredSum += tphSample**2
    dphSum += dphSample
    dphSquaredSum += dphSample**2
    vTSum += transverseVelSample
    v_ph_sum += vSample
 
    for idx in checkIndex:
        parallaxCheck[idx][i] = parallaxSample[idx]
        radVelCheck[idx][i] = radVelSample[idx]
        tphCheck[idx][i] = tphSample[idx]
        dphCheck[idx][i] = dphSample[idx]
 
#estimates
vT = vTSum / N #transeverse velecity
v_ph = v_ph_sum / N
tph = tphSum / N
tphError = np.sqrt(np.abs(tphSquaredSum / N - tph**2))
dph = dphSum / N
dphError = np.sqrt(np.abs(dphSquaredSum / N - dph**2))
 

for index in checkIndex:
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    fig.suptitle(f"Monte Carlo checking for Guassian distributions at source {index}", fontsize=12)
    
    #parallax
    axes[0, 0].hist(parallaxCheck[index], bins=50, color='darkgreen', edgecolor='black')
    axes[0, 0].axvline(parallax[index], color='red', linestyle='--', label='Measured value')
    axes[0, 0].set_xlabel("Parallax (mas)")
    axes[0, 0].set_ylabel("Count")
    axes[0, 0].set_title("Input: Parallax")
    axes[0, 0].legend()

    #radial velocity
    axes[0, 1].hist(radVelCheck[index], bins=50, color='darkgreen', edgecolor='black')
    axes[0, 1].axvline(radVel[index], color='red', linestyle='--', label='Measured value')
    axes[0, 1].set_xlabel("Radial Velocity (km/s)")
    axes[0, 1].set_ylabel("Count")
    axes[0, 1].set_title("Input: Radial Velocity")
    axes[0, 1].legend()

    #tph
    axes[1, 0].hist(tphCheck[index], bins=50, color='coral', edgecolor='black')
    axes[1, 0].axvline(tph[index], color='blue', linestyle='--', label='Mean')
    axes[1, 0].set_xlabel("$t_{ph}$ (yr)")
    axes[1, 0].set_ylabel("Count")
    axes[1, 0].set_title("Output: Perihelion Time $t_{ph}$")
    axes[1, 0].legend()

    #dph
    axes[1, 1].hist(dphCheck[index], bins=50, color='coral', edgecolor='black')
    axes[1, 1].axvline(dph[index], color='blue', linestyle='--', label='Mean')
    axes[1, 1].set_xlabel("$d_{ph}$ (pc)")
    axes[1, 1].set_ylabel("Count")
    axes[1, 1].set_title("Output: Perihelion Distance $d_{ph}$")
    axes[1, 1].legend()

    plt.tight_layout()
    plt.show()



#-------
#task 4

innerOort = 0.0097
outerOort = 0.48
proximaCenturi = 1.3

within10 = dph < 10

#without error bars
plt.figure(figsize=(8,6))
plt.scatter(dph[within10], tph[within10], s=5)

plt.xlabel("Perihelion Distance $d_{ph}$ (pc)")
plt.ylabel("Perihelion Time $t_{ph}$ (yr)")
plt.title("Stellar Flybys within 10 pc")
#oort cloud
plt.axvline(innerOort, linestyle='--', color = 'black', label='Inner Oort Cloud (~0.0097 pc)')
plt.axvline(outerOort, linestyle='--', color = 'black', label='Outer Oort Cloud (~0.48 pc)')
plt.axvline(proximaCenturi, linestyle='--', color = 'orange', label='Proxima Centrui (~1.3 pc)')
plt.legend()
plt.show()


#with error bars
#only show error bars if theyre <=50% of the measured value

goodErr10 = within10 & (dphError <= dph/2) & (tphError <= np.abs(tph)/2)

plt.figure(figsize=(8,6))
plt.scatter(dph[within10], tph[within10], s=5)
plt.errorbar(dph[goodErr10], tph[goodErr10], xerr=dphError[goodErr10], yerr=tphError[goodErr10], fmt='o', ms=3, color='steelblue', elinewidth=0.6, alpha = 0.5, capsize=2, zorder=1)

plt.xlabel("Perihelion Distance $d_{ph}$ (pc)")
plt.ylabel("Perihelion Time $t_{ph}$ (yr)")
plt.title("Stellar Flybys within 10 pc, error bars included")
#oort cloud
plt.axvline(innerOort, linestyle='--', color = 'black', label='Inner Oort Cloud (~0.0097 pc)')
plt.axvline(outerOort, linestyle='--', color = 'black', label='Outer Oort Cloud (~0.48 pc)')
plt.axvline(proximaCenturi, linestyle='--', color = 'orange', label='Proxima Centrui (~1.3 pc)')
plt.legend()
plt.show()

#############

within2 = dph < 2



plt.figure(figsize=(8,6))
plt.scatter(dph[within2], tph[within2], s=10)

plt.xlabel("Perihelion Distance $d_{ph}$ (pc)")
plt.ylabel("Perihelion Time $t_{ph}$ (yr)")
plt.title("Closest Stellar Flybys (within 2 pc)")
#oort cloud
plt.axvline(innerOort, linestyle='--', color = 'black', label='Inner Oort Cloud (~0.0097 pc)')
plt.axvline(outerOort, linestyle='--', color = 'black', label='Outer Oort Cloud (~0.48 pc)')
plt.axvline(proximaCenturi, linestyle='--', color = 'orange', label='Proxima Centrui (~1.3 pc)')
plt.legend()
plt.show()

#with error bars
#only show error bars if theyre <=100% of the measured value
goodErr2 = within2 & (dphError <= dph) & (tphError <= np.abs(tph))

plt.figure(figsize=(8,6))
plt.scatter(dph[within2], tph[within2], s=10)
plt.errorbar(dph[goodErr2], tph[goodErr2], xerr=dphError[goodErr2], yerr=tphError[goodErr2], fmt='o', ms=3, color='steelblue', elinewidth=0.6, capsize=2, zorder=1)

plt.xlabel("Perihelion Distance $d_{ph}$ (pc)")
plt.ylabel("Perihelion Time $t_{ph}$ (yr)")
plt.title("Closest Stellar Flybys (within 2 pc), error bars included")
#oort cloud
plt.axvline(innerOort, linestyle='--', color = 'black', label='Inner Oort Cloud (~0.0097 pc)')
plt.axvline(outerOort, linestyle='--', color = 'black', label='Outer Oort Cloud (~0.48 pc)')
plt.axvline(proximaCenturi, linestyle='--', color = 'orange', label='Proxima Centrui (~1.3 pc)')
plt.legend()
plt.show()

#-----
#task 5

#this is ± 5 million years, as present day is at the centre
within10Myr = np.abs(tph) < 5e6

numOuterOort = np.sum((dph < outerOort) & within10Myr)
numInnerOort = np.sum((dph < innerOort) & within10Myr)
numProxima = np.sum((dph < proximaCenturi) & within10Myr)

print("\nStars passing within Proxima Centauri distance:", numProxima)
print("Stars entering the outer Oort Cloud:", numOuterOort)
print("Stars entering the inner Oort Cloud:", numInnerOort)

#------
#task 6

#sort stars by closest perihelion distance
sortedIndices = np.argsort(dph)

#extract Gaia DR3 source ID
sourceID = data['SOURCE_ID_1']

print("\n10 closest stellar flybys:")
print("Source ID            |  Current distance (pc) |  tph (yr)                | dph (pc)")
print()
#lining all this up took so long
for i in range(10):
    idx = sortedIndices[i]
    
    print(f"{sourceID[idx]},    {d[idx]:.3f},                {tph[idx]:.3e} ± {tphError[idx]:.2e},        {dph[idx]:.3f} ± {dphError[idx]:.3f}")


#overplot the 10 closest approaching stars on the HR diagram
top10 = sortedIndices[:10]

#plot
plt.scatter(xAxis, M_G, s=1, color='black', label='All stars')
plt.scatter(xAxis[top10], M_G[top10], s=25, color='red',  label='10 closest flybys')
plt.gca().invert_yaxis()
plt.xlabel("BP - RP")
plt.ylabel("Absolute Magnitude $M_G$")
plt.title("H-R Diagram with 10 Closest Approaching Stars")
plt.legend()
plt.show()


#Part 2
print("\n---\nPart 2")
#task 1

#read in the file
mamajekData = []
with open('Mamajek_Relations.txt', 'r') as f:
    for line in f:
        line = line.strip()
        #this skips comment and header lines
        if line.startswith('#') or line == '':
            continue
        mamajekData.append(line.split())

TeffList = []
MsunList = []
BpRpList = []
'''originally tried doing this way, couldnt get it working
for row in mamajekData:
    #Teff is column 1
    if row[1] == '...':
        None
    else:
        teff = float(row[1])
        TeffList.append(teff)
   #Msun is column 30
    if row[30] == '...':
        None
    else:
        mass = float(row[30])
        MsunList.append(mass)
    #Bp-Rp is column 11    
    if row[11] == '...':
        None
    else:
        bprp = float(row[11])
        BpRpList.append(bprp)
doesnt work later on when trying to broadcast 2 operands of different shapes
''' 
for row in mamajekData:
    teff = np.nan if row[1]  in ['...', '.....'] else float(row[1])
    mass = np.nan if row[30] in ['...', '.....'] else float(row[30])
    bprp = np.nan if row[11] in ['...', '.....'] else float(row[11])
    TeffList.append(teff)
    MsunList.append(mass)
    BpRpList.append(bprp)

TeffArray = np.array(TeffList)
MsunArray = np.array(MsunList)
BpRpArray = np.array(BpRpList)

print(f"\nMamajek rows loaded: {len(TeffArray)}")
print(f"Valid Teff entries: {np.sum(~np.isnan(TeffArray))}")
print(f"Valid mass entries: {np.sum(~np.isnan(MsunArray))}")
print(f"Valid Bp-Rp entries: {np.sum(~np.isnan(BpRpArray))}")



#task 2
#Plot mass vs Teff and interpolate to get masses for GCNS stars


#sort data with Teff because interp1d needs the data to be continuously increasing
sortingIndex = np.argsort(TeffArray)
TeffSorted = TeffArray[sortingIndex]
massSorted = MsunArray[sortingIndex]

#plot of mass vs Teff
plt.figure(figsize=(8, 5))
plt.scatter(TeffSorted, massSorted, color='red', s=5, label='Mamajek sample')
plt.xlabel("Effective Temperature $T_{eff}$ (K)")
plt.ylabel("Mass ($M_\\odot$)")
plt.title("Mass vs Effective Temperature (Mamajek Relations)")
plt.legend()
plt.tight_layout()
plt.show()

#create the interpolation function, f(Teff) = M
Teff_to_MassFunction = interp1d(TeffSorted, massSorted, kind='linear', bounds_error=False, fill_value=np.nan)

#use on Gaia data that have a teff_gspphot value
teffGaia = data['teff_gspphot']

massFromInterpolation = Teff_to_MassFunction(teffGaia)

numWithTeff = np.sum(~np.isnan(teffGaia))
print(f"\nStars with Teff estimate: {numWithTeff}")


#task 3

num_without_teff = np.sum(np.isnan(teffGaia))
print(f"Stars without Teff estimate: {num_without_teff}\n")

#task 4



valid_bprp = ~np.isnan(BpRpArray) & ~np.isnan(MsunArray)
BpRpValid = BpRpArray[valid_bprp]
MassValidBpRp = MsunArray[valid_bprp]

#sort BP-RP for interp1d function
sortBpRp = np.argsort(BpRpValid)
BpRpSorted = BpRpValid[sortBpRp]
MassSortedBpRp = MassValidBpRp[sortBpRp]

#create interpolation function f(BP-RP) = M
BpRp_to_MassFunction = interp1d(BpRpSorted, MassSortedBpRp, kind='linear', bounds_error=False, fill_value=np.nan)

#estimate mass from BP-RP for stars missing Teff
massFromBpRp = BpRp_to_MassFunction(xAxis)

#this combines Teff mass and BP-RP mass
massEstimate = np.where(~np.isnan(massFromInterpolation), massFromInterpolation, massFromBpRp)

numBpRpUsed = np.sum(np.isnan(massFromInterpolation) & ~np.isnan(massFromBpRp))
numStillMissing = np.sum(np.isnan(massEstimate))
print(f"Stars from Teff interpolation: {np.sum(~np.isnan(massFromInterpolation))}")
print(f"Stars from BP-RP interpolation: {numBpRpUsed}")
print(f"Stars still without mass: {numStillMissing}")


#task 5
#calculate impulse transfer for each source

impulseValues = massEstimate / (v_ph * dph**2)

#print(f"\nImpulse values are: {impulseValues}") tester

plt.scatter(dph, impulseValues,color = 'red', s = 5)
plt.title("Impulse vs dph")
plt.xlabel("dph")
plt.ylabel("Impulse")
plt.show()

#task 6
#plot of tph vs dph with impulses

#focusing on flybys within 2 pc for the zoomed plot
closeFlybyDistance = dph < 2

#normalise impulse for symbol sizing
closeImpulses = impulseValues[closeFlybyDistance]
tphClose = tph[closeFlybyDistance]
dphClose = dph[closeFlybyDistance]


#normalise
#needed ai to help with the calculations in this part
impulse_finite = closeImpulses[np.isfinite(closeImpulses)]
i_min, i_max = np.nanmin(impulse_finite), np.nanmax(impulse_finite)
sizes = 5 + 195 * (closeImpulses - i_min) / (i_max - i_min)
sizes = np.nan_to_num(sizes, nan=5)

fig, ax = plt.subplots(figsize=(9, 6))

sc = ax.scatter(dphClose, tphClose, c=closeImpulses, s=sizes, cmap='plasma', norm=plt.matplotlib.colors.LogNorm())

cbar = plt.colorbar(sc, ax=ax)
cbar.set_label("Impulse", fontsize=10)

#reference lines
ax.axvline(innerOort, linestyle='--', color='black',label='Inner Oort Cloud (~0.0097 pc)')
ax.axvline(outerOort, linestyle='--', color='grey',label='Outer Oort Cloud (~0.48 pc)')
ax.axvline(proximaCenturi, linestyle='--', color='orange',label='Proxima Centauri (~1.3 pc)')
ax.axhline(0, linestyle='--', color='green', label='Present day')

ax.set_xlabel("Perihelion Distance $d_{ph}$ (pc)")
ax.set_ylabel("Perihelion Time $t_{ph}$ (yr)")
ax.set_title("Closest Stellar Flybys within 2 pc\n(symbol size and colour proportional impulse)")
ax.legend()
plt.show()

######

#with error bars
#only show error bars if there <=100% of the measured value
goodErrClose = closeFlybyDistance & (dphError <= dph) & (tphError <= np.abs(tph))

fig, ax = plt.subplots(figsize=(9, 6))

ax.errorbar(dph[goodErrClose], tph[goodErrClose], xerr=dphError[goodErrClose], yerr=tphError[goodErrClose], fmt='none', elinewidth=0.6, color='steelblue', zorder=0)
sc = ax.scatter(dphClose, tphClose, c=closeImpulses, s=sizes, cmap='plasma', norm=plt.matplotlib.colors.LogNorm())

cbar = plt.colorbar(sc, ax=ax)
cbar.set_label("Impulse", fontsize=10)


#reference lines
ax.axvline(innerOort, linestyle='--', color='black',label='Inner Oort Cloud (~0.0097 pc)')
ax.axvline(outerOort, linestyle='--', color='grey',label='Outer Oort Cloud (~0.48 pc)')
ax.axvline(proximaCenturi, linestyle='--', color='orange',label='Proxima Centauri (~1.3 pc)')
ax.axhline(0, linestyle='--', color='green', label='Present day')

ax.set_xlabel("Perihelion Distance $d_{ph}$ (pc)")
ax.set_ylabel("Perihelion Time $t_{ph}$ (yr)")
ax.set_title("Closest Stellar Flybys within 2 pc, error bars included\n(symbol size and colour proportional impulse)")
ax.legend()
plt.show()


#task 8


# Kepler-444 is in Gaia DR3, so i got the required data on kepler-444 manually and hardcoded it
#this was then used to calcualte the tph and dph

parallaxKepler444 = 27.4137 #mas
radVelKepler444 = -123.05 #km/s
pmRA_Kepler444 = 94.6824 #mas/yr
pmDec_Kepler444 = -632.202 # mas/yr

vT_Kepler444 = c2 * np.sqrt(pmRA_Kepler444**2 + pmDec_Kepler444**2) / parallaxKepler444
v_Kepler444  = np.sqrt(vT_Kepler444**2 + radVelKepler444**2)
tph_Kepler444 = -c1 * (1/parallaxKepler444) * (radVelKepler444 / v_Kepler444**2)
dph_Kepler444 =  1e3 * (1/parallaxKepler444) * (vT_Kepler444 / v_Kepler444)

print(f"\nKepler-444:  tph = {tph_Kepler444:.3e} yr,  dph = {dph_Kepler444:.2f} pc")
print("So it reaches its closest approach in 160,000 years, 24.38 pc from the Sun")





#part 3

#task 1
#read in the Ultracool Sheet
ultraCoolData = pd.read_csv('UltracoolSheet.csv')
numOfObjects = len(ultraCoolData)

print(f"\n---\nPart 3\nNumber of objects in Ultracool Sheet: {numOfObjects}")


#task 2
#absolute J vs J-K


#extract needed columns as numpy arrays
sptNum = pd.to_numeric(ultraCoolData['sptnum_formula'], errors='coerce').values
parallax_uc = pd.to_numeric(ultraCoolData['plx_formula'], errors='coerce').values  #parallax in mas
plxErr_uc = pd.to_numeric(ultraCoolData['plxerr_formula'], errors='coerce').values


#photometry for CMD
J_MKO = pd.to_numeric(ultraCoolData['J_MKO'], errors='coerce').values
K_MKO = pd.to_numeric(ultraCoolData['K_MKO'], errors='coerce').values


dist_uc = 1000 / parallax_uc
JminusK = J_MKO - K_MKO
M_J = J_MKO - 5 * np.log10(dist_uc) + 5

#sptnum_formula column
#from README section: M6=6, L0=10, T0=20
planetType = sptNum < 6
typeM = (sptNum >= 6) & (sptNum < 10)
typeL = (sptNum >= 10) & (sptNum < 20)
typeT = (sptNum >= 20) & (sptNum < 30)
typeY = sptNum >= 30


fig, ax = plt.subplots(figsize=(8, 8))

ax.scatter(JminusK[typeM], M_J[typeM],s=8, color='red', label='M dwarfs (M6+)')
ax.scatter(JminusK[typeL], M_J[typeL],s=8, color='darkred', label='L dwarfs')
ax.scatter(JminusK[typeT], M_J[typeT],s=8, color='skyblue', label='T dwarfs')
ax.scatter(JminusK[typeY], M_J[typeY],s=8, color='purple', label='Y dwarfs')
ax.scatter(JminusK[planetType], M_J[planetType],s=20, color='gold', edgecolors='black',linewidths=0.6, zorder=5, label='Imaged planets/companions')

ax.invert_yaxis()
ax.set_xlabel("J $-$ K (MKO)")
ax.set_ylabel("$M_J$")
ax.set_title("Colour Magnitude Diagram from Ultracool sheet")
ax.legend()
plt.tight_layout()
plt.show()

#task 3

#rv formula, pmdec formula,
#pmra formula, plx formula

#extract needed columns
name = ultraCoolData['name'].values
rv_uc = pd.to_numeric(ultraCoolData['rv_formula'], errors='coerce').values  #km/s
rvErr_uc = pd.to_numeric(ultraCoolData['rverr_formula'], errors='coerce').values
pmRA_uc = pd.to_numeric(ultraCoolData['pmra_formula'], errors='coerce').values  #mas/yr
pmraErr_uc = pd.to_numeric(ultraCoolData['pmraerr_formula'], errors='coerce').values
pmDec_uc= pd.to_numeric(ultraCoolData['pmdec_formula'], errors='coerce').values  #mas/yr
pmdecErr_uc= pd.to_numeric(ultraCoolData['pmdecerr_formula'], errors='coerce').values





#task 4

numWithRV_uc = np.sum(~np.isnan(rv_uc))
print(f"Sources with measured RV: {numWithRV_uc}")


#task 5
#reusing the equations from part 1, task 3

#calculating transverse velocity
vT_uc = c2 * (np.sqrt(pmRA_uc**2 + pmDec_uc**2)) / parallax_uc

#total veloctiy
v_ph_uc = np.sqrt(vT_uc**2 + rv_uc**2)

#perihelion time in years
tph_uc = -c1 * (1/parallax_uc) * (rv_uc / v_ph_uc**2)

#perihelion distance in pc
dph_uc = (1000/parallax_uc) * (vT_uc / v_ph_uc)

#plots, reused code from part 1

closestApproaches = dph_uc < 2

plt.scatter(dph_uc[closestApproaches], tph_uc[closestApproaches], s=10)

plt.xlabel("Perihelion Distance $d_{ph}$ (pc)")
plt.ylabel("Perihelion Time $t_{ph}$ (yr)")
plt.title("Closest Stellar Flybys within 2 pc")
#oort cloud
plt.axvline(innerOort, linestyle='--', color = 'black', label='Inner Oort Cloud (~0.0097 pc)')
plt.axvline(outerOort, linestyle='--', color = 'black', label='Outer Oort Cloud (~0.48 pc)')
plt.axvline(proximaCenturi, linestyle='--', color = 'orange', label='Proxima Centrui (~1.3 pc)')
plt.legend(loc='upper right')
plt.show()

#######

#MC error propagation for ultracool perihelion values, also same as Part 1

tphSum_uc = np.zeros(len(parallax_uc))
tphSquaredSum_uc = np.zeros(len(parallax_uc))
dphSum_uc = np.zeros(len(parallax_uc))
dphSquaredSum_uc = np.zeros(len(parallax_uc))

for i in range(N):
    sampleParallax = np.random.normal(parallax_uc, plxErr_uc)
    sampleRadVel = np.random.normal(rv_uc, rvErr_uc)
    sample_pmRA = np.random.normal(pmRA_uc, pmraErr_uc)
    sample_pmDec = np.random.normal(pmDec_uc, pmdecErr_uc)
    sample_vT = c2 * np.sqrt(sample_pmRA**2 + sample_pmDec**2) / sampleParallax
    sample_vph = np.sqrt(sample_vT**2 + sampleRadVel**2)
    sample_tph = -c1 * (1/sampleParallax) * (sampleRadVel / sample_vph**2)
    sample_dph = (1000/sampleParallax) * (sample_vT / sample_vph)
    tphSum_uc += sample_tph
    tphSquaredSum_uc += sample_tph**2
    dphSum_uc += sample_dph
    dphSquaredSum_uc += sample_dph**2

tph_ucError = np.sqrt(np.abs(tphSquaredSum_uc/N - (tphSum_uc/N)**2))
dph_ucError = np.sqrt(np.abs(dphSquaredSum_uc/N - (dphSum_uc/N)**2))

#plots , also reused code from part 1

#only show error bars if theyre <=100% of the measured value
goodErrUC = closestApproaches & (dph_ucError <= dph_uc) & (tph_ucError <= np.abs(tph_uc))

plt.scatter(dph_uc[closestApproaches], tph_uc[closestApproaches], s=10)
plt.errorbar(dph_uc[goodErrUC], tph_uc[goodErrUC], xerr=dph_ucError[goodErrUC], yerr=tph_ucError[goodErrUC], fmt='o', ms=3, color='steelblue', elinewidth=0.6, capsize=2, zorder=1)

plt.xlabel("Perihelion Distance $d_{ph}$ (pc)")
plt.ylabel("Perihelion Time $t_{ph}$ (yr)")
plt.title("Closest Stellar Flybys within 2 pc, error bars included")
#oort cloud
plt.axvline(innerOort, linestyle='--', color = 'black', label='Inner Oort Cloud (~0.0097 pc)')
plt.axvline(outerOort, linestyle='--', color = 'black', label='Outer Oort Cloud (~0.48 pc)')
plt.axvline(proximaCenturi, linestyle='--', color = 'orange', label='Proxima Centrui (~1.3 pc)')
plt.legend(loc='upper right')

plt.show()


#task 6
#find the closest approaching ultracool dwarf

validData = np.isfinite(dph_uc) & np.isfinite(tph_uc)
sortedUC  = np.argsort(dph_uc[validData]) #sorts the distances from smallest to largest
validIndex  = np.where(validData)[0] #index of closest distance


#closest approacher
closestIdx = validIndex[sortedUC[0]]
print(f"\nClosest ultracool dwarf: {name[closestIdx]}")
print(f"Perihelion distance: {dph_uc[closestIdx]:.4f} pc")
print(f"Perihelion time: {tph_uc[closestIdx]:.3e} yr")
print(f"Spectral type number: {sptNum[closestIdx]}\n\n")

#10 closest appraochers to be compared to Gaia data
print("\n10 closest ultracool flybys:\n")
for i in range(10):
    index = validIndex[sortedUC[i]]
    print(f"Name: {name[index]}\ndph (pc): {dph_uc[index]:.4f}\ntph (yr): {tph_uc[index]:.4f}\nsptnum: {sptNum[index]}\n")
#need to compare these to the values printed above in part 1

#same plot as earlier with closest approacher overplotted
fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(JminusK[typeM], M_J[typeM],s=10, color='red', label='M dwarfs (M6+)')
ax.scatter(JminusK[typeL], M_J[typeL],s=10, color='darkred', label='L dwarfs')
ax.scatter(JminusK[typeT], M_J[typeT],s=10, color='skyblue', label='T dwarfs')
ax.scatter(JminusK[typeY], M_J[typeY],s=10, color='purple', label='Y dwarfs')
ax.scatter(JminusK[planetType], M_J[planetType],s=20, color='gold', edgecolors='black',linewidths=0.6, label='Companions')

#overplot the closest approacher
ax.scatter(JminusK[closestIdx], M_J[closestIdx], s=100, color='lime', edgecolors='black',linewidths=1.2,label='Closest approacher')

ax.invert_yaxis()
ax.set_xlabel("J $-$ K (MKO)")
ax.set_ylabel("$M_J$")
ax.set_title("Same plot with closest approaching Ultracool Dwarf highlighted")
ax.legend(loc='upper left',fontsize=12)
plt.tight_layout()
plt.show()

print("\nDue to its position on the plot, the closest approacher is most likely an L dwarf")