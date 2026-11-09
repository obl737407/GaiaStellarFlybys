#import libraries for part 1;
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
#part 2;
from scipy.interpolate import interp1d
#part 3;
import pandas as pd

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
print(data.columns) #just to visualise


#plt.errorbar(x, y, kwargs) (radial_velocity,)

#extracting the needed columns
parallax = data['PARALLAX_1']
parallaxError = data['PARALLAX_ERROR_1']
bp = data['PHOT_BP_MEAN_MAG_1']

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

#not sure if i need the following
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

#put errors into BP-RP and M_G

colourError = np.sqrt(bpError**2 + rpError**2)
M_G_Error = np.sqrt(gError**2 + ((5/np.log(10)) * parallaxError / parallax)**2)

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
    pmRA_sample = np.random.normal(properMotionRA, properMotionRA_Error)
    pmDecSample = np.random.normal(properMotionDec, properMotionDec_Error)
 
    #apply the same equations to each sample
    transverseVelSample = c2 * np.sqrt(pmRA_sample**2 + pmDecSample**2) / parallaxSample
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

plt.figure(figsize=(8,6))
plt.errorbar(dph[within10], tph[within10], xerr=dphError[within10], yerr=tphError[within10], fmt='o', color='steelblue')

plt.xlabel("Perihelion Distance $d_{ph}$ (pc)")
plt.ylabel("Perihelion Time $t_{ph}$ (yr)")
plt.title("Stellar Flybys within 10 pc")
#oort cloud
plt.axvline(innerOort, linestyle='--', color = 'black', label='Inner Oort Cloud (~0.0097 pc)')
plt.axvline(outerOort, linestyle='--', color = 'black', label='Outer Oort Cloud (~0.48 pc)')
plt.axvline(proximaCenturi, linestyle='--', color = 'orange', label='Proxima Centrui (~1.3 pc)')
plt.legend()

plt.show()



within2 = dph < 2

plt.figure(figsize=(8,6))
plt.errorbar(dph[within2], tph[within2],
             xerr=dphError[within2], yerr=tphError[within2],
             fmt='o', ms=3, elinewidth=0.4,  capsize=0, color='steelblue')

plt.xlabel("Perihelion Distance $d_{ph}$ (pc)")
plt.ylabel("Perihelion Time $t_{ph}$ (yr)")
plt.title("Closest Stellar Flybys (within 2 pc)")
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
print("Source ID            |  Current dist (pc) |  tph (yr)               | dph (pc)")
print()
for i in range(10):
    idx = sortedIndices[i]
    print(f"{sourceID[idx]},    {d[idx]:.3f},    {tph[idx]:.3e} ± {tphError[idx]:.2e},    {dph[idx]:.3f} ± {dphError[idx]:.3f}")


#overplot the 10 closest approaching stars on the HR diagram
top10 = sortedIndices[:10]

#plot
plt.scatter(xAxis, M_G, s=1, color='black', label='All stars')
#error bars only on the 10 highlighted stars (adding them to all ~100k would be unreadable)
plt.errorbar(xAxis[top10], M_G[top10], xerr=colourError[top10], yerr=M_G_Error[top10], fmt='o', color='red',  label='10 closest flybys')
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

#sort by BP-RP for interp1d
sortBpRp = np.argsort(BpRpValid)
BpRpSorted = BpRpValid[sortBpRp]
MassSortedBpRp = MassValidBpRp[sortBpRp]

#create interpolation function f(BP-RP) = M
BpRp_to_MassFunction = interp1d(BpRpSorted, MassSortedBpRp, kind='linear', bounds_error=False, fill_value=np.nan)

#calculate BP-RP colour for GCNS stars
bpRpGaia = bp - rp

#estimate mass from BP-RP for stars missing Teff
massFromBpRp = BpRp_to_MassFunction(bpRpGaia)

#this combines Teff-based mass and BP-RP mass
massEstimate = np.where(~np.isnan(massFromInterpolation), massFromInterpolation, massFromBpRp)

numBpRpUsed = np.sum(np.isnan(massFromInterpolation) & ~np.isnan(massFromBpRp))
numStillMissing = np.sum(np.isnan(massEstimate))
print(f"Masses from Teff interpolation: {np.sum(~np.isnan(massFromInterpolation))}")
print(f"Masses from BP-RP interpolation: {numBpRpUsed}")
print(f"Stars still without mass: {numStillMissing}")

#task 5
#calculate impulse transfer for each source

impulseValues = massEstimate / (v_ph * dph**2)

#print(f"\nImpulse values are: {impulseValues}") tester

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

#draw error bars first so they sit underneath the scatter points
ax.errorbar(dphClose, tphClose, xerr=dphError[closeFlybyDistance], yerr=tphError[closeFlybyDistance], fmt='none',  color='grey',)

sc = ax.scatter(dphClose, tphClose, c=closeImpulses, s=sizes, cmap='plasma', norm=plt.matplotlib.colors.LogNorm(), zorder=1)

cbar = plt.colorbar(sc, ax=ax)
cbar.set_label("Impulse", fontsize=10)


#reference lines
ax.axvline(innerOort, linestyle='--', color='black',label='Inner Oort Cloud (~0.0097 pc)')
ax.axvline(outerOort, linestyle='--', color='grey',label='Outer Oort Cloud (~0.48 pc)')
ax.axvline(proximaCenturi, linestyle='--', color='orange',label='Proxima Centauri (~1.3 pc)')
ax.axhline(0, linestyle='--', color='green', label='Present day')

ax.set_xlabel("Perihelion Distance $d_{ph}$ (pc)")
ax.set_ylabel("Perihelion Time $t_{ph}$ (yr)")
ax.set_title("Closest Stellar Flybys within 2 pc\n(symbol size & colour ∝ impulse)")
ax.legend()
plt.show()


#part 3

#task 1
#read in the Ultracool Sheet
ultraCoolData = pd.read_csv('UltracoolSheet.csv')
numOfObjects = len(ultraCoolData)

print(f"\n---\nPart 3\nNumber of objects in Ultracool Sheet: {numOfObjects}")


#task 2
#absolute J vs J-K


#extract needed columns as numpy arrays
sptNum = pd.to_numeric(ultraCoolData['sptnum_formula'],   errors='coerce').values
plx_uc = pd.to_numeric(ultraCoolData['plx_formula'],      errors='coerce').values  # parallax (mas)
plxErr_uc = pd.to_numeric(ultraCoolData['plxerr_formula'],   errors='coerce').values


#photometry for CMD
J_MKO = pd.to_numeric(ultraCoolData['J_MKO'],   errors='coerce').values
K_MKO = pd.to_numeric(ultraCoolData['K_MKO'],   errors='coerce').values


dist_uc = 1000 / plx_uc
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
pmra_uc = pd.to_numeric(ultraCoolData['pmra_formula'], errors='coerce').values  #mas/yr
pmraErr_uc = pd.to_numeric(ultraCoolData['pmraerr_formula'], errors='coerce').values
pmdec_uc= pd.to_numeric(ultraCoolData['pmdec_formula'], errors='coerce').values  #mas/yr
pmdecErr_uc= pd.to_numeric(ultraCoolData['pmdecerr_formula'], errors='coerce').values





#task 4

numWithRV_uc = np.sum(~np.isnan(rv_uc))
print(f"Sources with measured RV: {numWithRV_uc}")


#task 5
#reusing the equations from part 1, task 3

#calculating transverse velocity
vT_uc = c2 * (np.sqrt(pmra_uc**2 + pmdec_uc**2)) / plx_uc

#total veloctiy
v_ph_uc = np.sqrt(vT_uc**2 + rv_uc**2)

#perihelion time in years
tph_uc = -c1 * (1/plx_uc) * (rv_uc / v_ph_uc**2)

#perihelion distance in pc
dph_uc = (1000/plx_uc) * (vT_uc / v_ph_uc)


#MC error propagation for ultracool perihelion values (same method as Part 1)
N_uc = 1000
tph_uc_sum    = np.zeros(len(plx_uc))
tph_uc_sq_sum = np.zeros(len(plx_uc))
dph_uc_sum    = np.zeros(len(plx_uc))
dph_uc_sq_sum = np.zeros(len(plx_uc))

for i in range(N_uc):
    plx_s   = np.random.normal(plx_uc,   plxErr_uc)
    rv_s    = np.random.normal(rv_uc,    rvErr_uc)
    pmra_s  = np.random.normal(pmra_uc,  pmraErr_uc)
    pmdec_s = np.random.normal(pmdec_uc, pmdecErr_uc)
    vT_s    = c2 * np.sqrt(pmra_s**2 + pmdec_s**2) / plx_s
    vph_s   = np.sqrt(vT_s**2 + rv_s**2)
    tph_s   = -c1 * (1/plx_s) * (rv_s / vph_s**2)
    dph_s   = (1000/plx_s) * (vT_s / vph_s)
    tph_uc_sum    += tph_s
    tph_uc_sq_sum += tph_s**2
    dph_uc_sum    += dph_s
    dph_uc_sq_sum += dph_s**2

tph_uc_err = np.sqrt(np.abs(tph_uc_sq_sum/N_uc - (tph_uc_sum/N_uc)**2))
dph_uc_err = np.sqrt(np.abs(dph_uc_sq_sum/N_uc - (dph_uc_sum/N_uc)**2))

#plots , also reused code from part 1

closestApproaches = dph_uc < 2

plt.figure(figsize=(8,6))
plt.errorbar(dph_uc[closestApproaches], tph_uc[closestApproaches], xerr=dph_uc_err[closestApproaches], yerr=tph_uc_err[closestApproaches], fmt='o',  color='steelblue')

plt.xlabel("Perihelion Distance $d_{ph}$ (pc)")
plt.ylabel("Perihelion Time $t_{ph}$ (yr)")
plt.title("Closest Stellar Flybys within 2 pc")
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