print("\rbip...", end="")

import numpy as np
from qutip import tensor, qeye, sigmax, sigmay, sigmaz
#from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Rectangle
import time

print("\rbip boup...")

GLOBAL_PRINT = True


### - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def get_tags(base, size, wmax=-1):
    wmax %= size+1
    if wmax == 0: return np.array([0])
    if size == 1: return np.arange(base)
    tags_0 = get_tags(base, size-1, min(size-1,wmax))
    tags_1 = get_tags(base, size-1, wmax-1)
    tags = tags_0*base
    for b in range(1,base):
        tags = np.append(tags, tags_1*base + b)
    return tags


def letterify(tag, size): # base 4
    string = ""
    for _ in range(size):
        match tag & 3:
            case 0: string = "I" + string
            case 1: string = "X" + string
            case 2: string = "Z" + string
            case 3: string = "Y" + string
        tag >>= 2
    return string


def binrepr(tag, size): # base 2
    string = ""
    for _ in range(size):
        match tag & 1:
            case 0: string = "0" + string
            case 1: string = "1" + string
        tag >>= 1
    return string


@np.vectorize(excluded=("g","gc"))
def get_IXYZ_error_proba(t, g=1, gc=1):
    pX = (1-np.exp(-2*g*t))/4
    pY = pX
    pZ = (1-np.exp(-(g+gc)*t))/2 - pX
    pI = 1-pX-pY-pZ
    return pI, pX, pY, pZ


@np.vectorize(signature="()->()")
def entropy_form(p):
    if 0 < p < 1: return - p*np.log(p)
    return 0


@np.vectorize(signature="(),()->()")
def get_weight(err, coden):
    w = coden
    for _ in range(coden):
        w -= (err & 3) == 0
        err >>= 2
    return w


### - - - - - - - - - - - - - - - - - - - - - - -


class QEC_code:
    def __init__(self, name, code_string):
        self.name = name
        self.code_string = code_string
        self.n = len(code_string[0])
        self.m = len(code_string)
        self.set_code()
        self.compute_stabilizers()

    def set_code(self):
        self.code = []
        for stabilizer in self.code_string:
            stab_string = ""
            for i,M in enumerate(stabilizer):
                match M:
                    case "I" | "0": stab_string += "0"
                    case "X" | "1": stab_string += "1"
                    case "Y" | "3": stab_string += "3"
                    case "Z" | "2": stab_string += "2"
            self.code.append(int(stab_string,4)) 
            # stabilisers generators are stored as base 4 int representing its associated Pauli sequence

    def compute_stabilizers(self):
        stabilizers = []
        tags = get_tags(2,self.m)
        for tag in tags:
            stab = 0
            for i in range(self.m):
                stab ^= self.code[i] * (tag & 1)
                tag >>= 1
            stabilizers.append(stab)
        self.stabilizers = list(set(stabilizers))


NQSC_code = QEC_code("NQSC", [
    "ZZIIIIIII",    
    "IZZIIIIII",
    "IIIZZIIII",
    "IIIIZZIII",
    "IIIIIIZZI",
    "IIIIIIIZZ",
    "XXXXXXIII",
    "IIIXXXXXX"
])

SQSC_code = QEC_code("SQSC", [
    "IIIXXXX",
    "IXXIIXX",
    "XIXIXIX",
    "IIIZZZZ",
    "IZZIIZZ",
    "ZIZIZIZ"
])

LMPZ_code = QEC_code("LMPZ", [
    "XZZXI",
    "IXZZX",
    "XIXZZ",
    "ZXIXZ"
])

X_code = QEC_code("X", [
    "IZZ",
    "ZZI"
])

Z_code = QEC_code("Z", [
    "IXX",
    "XXI"
])

def rotated_surface_code(d):
    if d % 2 == 0: raise ValueError
    code = []
    for y in range(d-1): # bulk 4qubits
        for x in range(d-1): 
            stab = ["I"]*d**2
            op = "Z" if (x+y) % 2 == 0 else "X"
            stab[x  +d*y  ] = op
            stab[x+1+d*y  ] = op
            stab[x  +d*y+d] = op
            stab[x+1+d*y+d] = op
            code.append("".join(stab))
    for i in range(0,d-1,2): # outer 2qubits # up
        stab = ["I"]*d**2
        stab[i] = "X"
        stab[i+1] = "X"
        code.append("".join(stab))
    for i in range(0,d-1,2): # right
        stab = ["I"]*d**2
        stab[(i+1)*d-1] = "Z"
        stab[(i+2)*d-1] = "Z"
        code.append("".join(stab))
    for i in range(0,d-1,2): # down
        stab = ["I"]*d**2
        stab[-1-i] = "X"
        stab[-2-i] = "X"
        code.append("".join(stab))
    for i in range(0,d-1,2): # left
        stab = ["I"]*d**2
        stab[-i*d-1] = "Z"
        stab[-(i+1)*d-1] = "Z"
        code.append("".join(stab))
    return QEC_code(f"{d}RSC", code)
    

### - - - - - - - - - - - - - - - - - - - - - - -


@np.vectorize(signature="(),(4),()->()") 
def get_proba(err, p_unit, coden):
    _err = err
    p = 1
    for _ in range(coden):
        p *= p_unit[err&3]
        err >>= 2
    if GLOBAL_PRINT: print(f"get_proba: {letterify(_err,coden)} -> p = {p}")
    return p


@np.vectorize(signature="(),(n),()->()") 
def get_syndrome(err, stab_gen, coden):
    syndrome = 0
    for S in stab_gen:
        E = err
        trigger = 0
        for _ in range(coden):
            x = E & 3
            y = S & 3
            trigger ^= not (x == 0 or y == 0 or x == y)
            E >>= 2
            S >>= 2
        syndrome = (syndrome << 1) + trigger
    if GLOBAL_PRINT: print(f"get_syndrome: {letterify(err,coden)} -> s = {binrepr(syndrome,coden-1)}")
    return syndrome


def find_syndrome_errors(error_syndromes, syndrome_tags, syndrome_ind):
    syndrome_errors = [[] for _ in syndrome_tags]
    for i,s in enumerate(error_syndromes):
        syndrome_errors[syndrome_ind[s]].append(i)
        if GLOBAL_PRINT: print(f"find_syndrome_errors: {i+1}/{len(error_syndromes)}")
    return syndrome_errors


def get_syndrome_proba(syndrome_errors, error_proba):
    syndrome_proba = [0]*len(syndrome_errors)
    for i,L in enumerate(syndrome_errors):
        for j in L:
            syndrome_proba[i] += error_proba[j]
        if GLOBAL_PRINT: print(f"get_syndrome_proba: {i+1}/{len(syndrome_errors)}")
    return syndrome_proba


def sorted_syndrome_error_indices(syndrome_errors, error_proba, syndrome_proba=None):
    if syndrome_proba is None: syndrome_proba = get_syndrome_proba(syndrome_errors, error_proba) 
    syndrome_ind = sorted(range(len(syndrome_errors)), reverse=True, key=lambda i: syndrome_proba[i])
    syndrome_error_ind = [sorted(syndrome_errors[si], reverse=True, key=lambda i: error_proba[i]) for si in range(len(syndrome_errors))]
    return syndrome_ind, syndrome_error_ind


def apply_correction(syndrome_errors, error_proba, error_tags, coden):
    corrected = [None]*len(error_tags)
    for si, errors in enumerate(syndrome_errors):
        if GLOBAL_PRINT: print(f"apply_correction: {si}/{len(syndrome_errors)}")
        mpe_proba = 0
        mpe_tag = None
        for ei in errors:
            if error_proba[ei] > mpe_proba:
                mpe_proba = error_proba[ei]
                mpe_tag = error_tags[ei]
        for ei in errors:
            corrected[ei] = error_tags[ei] ^ mpe_tag
            #print(f"| {letterify(error_tags[ei],coden)} x {letterify(mpe_tag,coden)} = {letterify(corrected[ei],coden)}")
    return corrected
            

def get_new_proba(error_proba, corrected_errors, corrected_error_tags):
    new_proba = [0]*len(corrected_error_tags)
    for ei,nei in enumerate(corrected_errors):
        if GLOBAL_PRINT: print(f"get_new_proba: {ei}/{len(corrected_errors)}")
        new_proba[nei] += error_proba[ei]
    return new_proba


def get_logical_error_probability(error_proba, error_tags, stabilizers):
    return np.sum([p for p,t in zip(error_proba,error_tags) if t in stabilizers])


def compute_threashold(p,p_prime):
    y = np.array(p_prime) - np.array(p)
    if y[0] > 0: return 0
    for i in range(1, len(y)):
        if y[i] > 0:
            return p[i-1] - y[i-1] * (p[i]-p[i-1]) / (y[i]-y[i-1])
        

### - - - - - - - - - - - - - - - - - - - - - - -


def plot_errors_to_syndromes(syndrome_errors, error_proba, syndrome_proba, error_tags, syndrome_tags, code, syndrome_ind=None, syndrome_errors_ind=None,
                             fuse_errors=False, colormap=cm.coolwarm, title=None, figsize=None, show_error_tags=True, error_tag_lim=None, highlight_stabilizers=False):
    if syndrome_ind is None or syndrome_errors_ind is None: syndrome_ind, syndrome_errors_ind = sorted_syndrome_error_indices(syndrome_errors, error_proba, syndrome_proba)
    if figsize is None: figsize = (max(20,len(syndrome_tags))/2, max(9,9 + np.log(len(syndrome_tags)/64)))
    if error_tag_lim is None: error_tag_lim = 0.02*figsize[1]/9
    fig,ax = plt.subplots(figsize=figsize)
    for s_pos,si in enumerate(syndrome_ind):
        print(f"plotting: {s_pos}/{len(syndrome_ind)}")
        errors_ind = syndrome_errors_ind[si]
        w0 = None
        heights = [0]
        colors = []
        tags = []
        for i in range(len(errors_ind)):
            weight = get_weight(error_tags[errors_ind[i]],code.n)
            if fuse_errors and weight == w0:
                heights[-1] += error_proba[errors_ind[i]]
                tags[-1] = -1
            else:
                heights.append(heights[-1] + error_proba[errors_ind[i]])
                colors.append(colormap( weight/(code.n+0.25) + (0 if fuse_errors else np.random.random()*0.25/(code.n+0.25)) ))
                tags.append(error_tags[errors_ind[i]])
                w0 = weight
        for i in range(len(heights)-1):
            h = heights[i+1]-heights[i]
            ax.add_patch(Rectangle((s_pos-0.4, heights[i]), 0.8, h, facecolor=colors[i]))
            if show_error_tags and tags[i] != -1 and h/syndrome_proba[syndrome_ind[0]] > error_tag_lim:
                if highlight_stabilizers and tags[i] in code.code:
                    ax.text(s_pos, heights[i]+h/2, letterify(tags[i],code.n), ha="center", va="center", color="black", fontfamily="monospace", fontsize=6, fontweight="bold")
                elif highlight_stabilizers and tags[i] in code.stabilizers:
                    ax.text(s_pos, heights[i]+h/2, letterify(tags[i],code.n), ha="center", va="center", color="black", fontfamily="monospace", fontsize=6)
                else : 
                    ax.text(s_pos, heights[i]+h/2, letterify(tags[i],code.n), ha="center", va="center", color="white", fontfamily="monospace", fontsize=6)
    ax.set_xlim(-0.5, len(syndrome_tags)-0.5)
    ax.set_ylim(0, syndrome_proba[syndrome_ind[0]])
    ax.set_xticks(range(len(syndrome_tags)))
    ax.set_xticklabels([binrepr(s,code.m) for s in syndrome_tags], rotation=90)
    if not title is None: fig.suptitle(title)
    plt.tight_layout()


### - - - - - - - - - - - - - - - - - - - - - - -


def bidule():
    code = rotated_surface_code(3)
    file = f"saves/{code.name}"
    wmax = code.n - 0
    g, gc = 1, 1
    t = 1

    IXYZ_proba = get_IXYZ_error_proba(t,g,gc)

    error_tags = get_tags(4, code.n, wmax)
    error_ind = {e:i for i,e in enumerate(error_tags)}
    syndrome_tags = get_tags(2, code.m)
    syndrome_ind = {s:i for i,s in enumerate(syndrome_tags)}

    error_proba = get_proba(error_tags, IXYZ_proba, code.n)
    error_syndromes = get_syndrome(error_tags, code.code, code.n) # array of syndromes
    error_syndromes = np.array([syndrome_ind[s] for s in error_syndromes]) # array of indices of syndromes

    syndrome_errors = find_syndrome_errors(error_syndromes, syndrome_tags, syndrome_ind) # array in indices again
    syndrome_proba = get_syndrome_proba(syndrome_errors, error_proba) 

    sorted_syndrome_ind, sorted_syndrome_errors_ind = sorted_syndrome_error_indices(syndrome_errors, error_proba, syndrome_proba)

    corrected_errors = apply_correction(syndrome_errors, error_proba, error_tags, code.n) # list of tags
    corrected_error_tags = list(set(corrected_errors)) # <- useful as wmax <= code.n
    corrected_error_ind = {e:i for i,e in enumerate(corrected_error_tags)}
    corrected_errors = [corrected_error_ind[corrected_errors[ei]] for ei in range(len(error_tags))] # list of indices

    corrected_error_proba = get_new_proba(error_proba, corrected_errors, corrected_error_tags)

    fidelity = get_logical_error_probability(corrected_error_proba, corrected_error_tags, code.stabilizers)
    plot_errors_to_syndromes(syndrome_errors, error_proba, syndrome_proba, error_tags, syndrome_tags, code, 
                             sorted_syndrome_ind, sorted_syndrome_errors_ind, fuse_errors=False, show_error_tags=True, highlight_stabilizers=True, error_tag_lim=0, figsize=(300,100),
                             title=f"{code.name} (approx {code.n-wmax})  -  g = {g}  -  gc = {gc}  -  t = {t:.3e}  -  p = {round(( 1-IXYZ_proba[0] )*100,2)}%  -  p' = {round(( 1-fidelity )*100,2)}%")
    
    data_entropy = np.sum(entropy_form(error_proba))
    measure_entropy = np.sum(entropy_form(syndrome_proba))
    corrected_data_entropy = np.sum(entropy_form(corrected_error_proba))

    plt.savefig(f"{file}/bigassplot.png")
    

### - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def truc():
    code = rotated_surface_code(3)
    file = f"saves/{code.name}"
    wmax = code.n - 0
    g, gc = 1, 1
    times = np.logspace(-3,0.5,22) # 8 15 22 29 36 43 50

    param = np.array((code, wmax, g, gc, times), dtype=object)
    np.save(f"{file}/param", param)

    data_entropies = np.array([])
    measure_entropies = np.array([])
    corrected_data_entropies = np.array([])
    error_probas = np.array([])
    fidelities = np.array([])

    for i,t in enumerate(times):

        IXYZ_proba = get_IXYZ_error_proba(t,g,gc)

        error_tags = get_tags(4, code.n, wmax)
        error_ind = {e:i for i,e in enumerate(error_tags)}
        syndrome_tags = get_tags(2, code.m)
        syndrome_ind = {s:i for i,s in enumerate(syndrome_tags)}

        error_proba = get_proba(error_tags, IXYZ_proba, code.n)
        error_syndromes = get_syndrome(error_tags, code.code, code.n) # array of syndromes
        error_syndromes = np.array([syndrome_ind[s] for s in error_syndromes]) # array of indices of syndromes

        syndrome_errors = find_syndrome_errors(error_syndromes, syndrome_tags, syndrome_ind) # array in indices again
        syndrome_proba = get_syndrome_proba(syndrome_errors, error_proba) 

        sorted_syndrome_ind, sorted_syndrome_errors_ind = sorted_syndrome_error_indices(syndrome_errors, error_proba, syndrome_proba)

        corrected_errors = apply_correction(syndrome_errors, error_proba, error_tags, code.n) # list of tags
        corrected_error_tags = list(set(corrected_errors)) # <- useful as wmax <= code.n
        corrected_error_ind = {e:i for i,e in enumerate(corrected_error_tags)}
        corrected_errors = [corrected_error_ind[corrected_errors[ei]] for ei in range(len(error_tags))] # list of indices

        corrected_error_proba = get_new_proba(error_proba, corrected_errors, corrected_error_tags)

        fidelity = get_logical_error_probability(corrected_error_proba, corrected_error_tags, code.stabilizers)
        plot_errors_to_syndromes(syndrome_errors, error_proba, syndrome_proba, error_tags, syndrome_tags, code, 
                                 sorted_syndrome_ind, sorted_syndrome_errors_ind, fuse_errors=True, show_error_tags=False, highlight_stabilizers=False,
                                 title=f"{code.name} (approx {code.n-wmax})  -  g = {g}  -  gc = {gc}  -  t = {t:.3e}  -  p = {round(( 1-IXYZ_proba[0] )*100,2)}%  -  p' = {round(( 1-fidelity )*100,2)}%")
        
        data_entropy = np.sum(entropy_form(error_proba))
        measure_entropy = np.sum(entropy_form(syndrome_proba))
        corrected_data_entropy = np.sum(entropy_form(corrected_error_proba))

        plt.savefig(f"{file}/{str(i).zfill(3)}.png")
        plt.close()
        data_entropies =           np.append(data_entropies,           [data_entropy])
        measure_entropies =        np.append(measure_entropies,        [measure_entropy])
        corrected_data_entropies = np.append(corrected_data_entropies, [corrected_data_entropy])
        error_probas =             np.append(error_probas,             [1-IXYZ_proba[0]])
        fidelities =               np.append(fidelities,               [fidelity])
        np.save(f"{file}/data", (data_entropies, measure_entropies, corrected_data_entropies, error_probas, fidelities))


### - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


#def plotplot():
#    file="saves"
#
#    code, wmax, g, gc, times = np.load(f"{file}/param.npy", allow_pickle=True)
#    data_entropies, measure_entropies, corrected_data_entropies, error_probas, fidelities = [np.array(L) for L in np.load(f"{file}/data.npy")]
#    efficacities = (data_entropies - corrected_data_entropies)/measure_entropies
#
#    threashold = compute_threashold(error_probas, 1-fidelities)
#
#    fig, ax = plt.subplots(2,2,figsize=(24,12))
#    fig.suptitle(f"{code.name} (approx {code.n-wmax})  -  g = {g}  -  gc = {gc}")
#
#    ax[0,0].loglog(times, data_entropies, lw=2, c="crimson", label="data")
#    ax[0,0].loglog(times, measure_entropies, lw=2, c="darkslateblue", label="measurement")
#    ax[0,0].loglog(times, corrected_data_entropies, lw=2, c="darkorange", label="corrected data")
#    ax[0,0].legend()
#    ax[0,0].set_title("entropy")
#    ax[0,0].grid(which='minor', linewidth=0.3, alpha=0.5)
#
#    ax[0,1].loglog(times, efficacities, lw=2, c="goldenrod")
#    ax[0,1].set_title("efficacity")
#    ax[0,1].grid(which='minor', linewidth=0.3, alpha=0.5)
#
#    ax[1,0].loglog(times, error_probas, lw=2, c="crimson", label="p : before correction")
#    ax[1,0].loglog(times, 1-fidelities, lw=2, c="darkorange", label="p' : after correction (fidelity)")
#    ax[1,0].grid(which='minor', linewidth=0.3, alpha=0.5)
#    ax[1,0].legend()
#    ax[1,0].set_title("logical error probability")
#
#    ax[1,1].loglog(*([[0,error_probas[-1]**(1/code.n)]]*2), lw=1.5, ls="--", c="lightgrey", alpha=0.5)
#    ax[1,1].loglog(error_probas, 1-fidelities, lw=2, c="teal")
#    ax[1,1].grid(which='minor', linewidth=0.3, alpha=0.5)
#    ax[1,1].set_title("p vs p'" if threashold is None else f"p vs p' (p_th = {round(threashold*100,2)}%)")
#
#    plt.tight_layout()
#    plt.show()



def plotplot():
    file="saves/LMPZ"

    code, wmax, g, gc, times = np.load(f"{file}/param.npy", allow_pickle=True)
    data_entropies, measure_entropies, corrected_data_entropies, error_probas, fidelities = [np.array(L) for L in np.load(f"{file}/data.npy")]
    efficacities = (data_entropies - corrected_data_entropies)/measure_entropies

    threashold = compute_threashold(error_probas, 1-fidelities)

    fig, ax = plt.subplots(1,3,figsize=(26,9))
    fig.suptitle(f"{code.name} (approx {code.n-wmax})  -  g = {g}  -  gc = {gc}")

    ax[0].loglog(error_probas, data_entropies, lw=2, c="crimson", label="data")
    ax[0].loglog(error_probas, corrected_data_entropies, lw=2, c="deeppink", label="corrected data")
    ax[0].loglog(error_probas, measure_entropies, lw=2, c="mediumseagreen", label="measurement")
    ax[0].legend()
    ax[0].set_title("entropy")
    ax[0].grid(which='minor', linewidth=0.3, alpha=0.5)

    ax[1].loglog([0,error_probas[-1]**(1/code.n)], [0,error_probas[-1]**(1/code.n)], lw=1.5, ls="--", c="lightgrey", alpha=0.5)
    ax[1].loglog(error_probas, 1-fidelities, lw=2, c="mediumseagreen")
    ax[1].grid(which='minor', linewidth=0.3, alpha=0.5)
    ax[1].set_title("logical error probability" + ("" if threashold is None else f" (p_th = {round(threashold*100,2)}%)"))

    ax[2].loglog(error_probas, efficacities, lw=2, c="mediumseagreen")
    ax[2].set_title("efficacity")
    ax[2].grid(which='minor', linewidth=0.3, alpha=0.5)
    
    if not threashold in (None, 0): 
        ax[0].scatter([threashold], np.interp([threashold],error_probas,data_entropies),         s=40, c="crimson")
        ax[0].scatter([threashold], np.interp([threashold],error_probas,measure_entropies),       s=40, c="mediumseagreen")
        ax[0].scatter([threashold], np.interp([threashold],error_probas,corrected_data_entropies), s=40, c="deeppink")
        ax[1].scatter([threashold], np.interp([threashold],error_probas,1-fidelities),              s=40, c="mediumseagreen")
        ax[2].scatter([threashold], np.interp([threashold],error_probas,efficacities),               s=40, c="mediumseagreen")
    
    plt.tight_layout()
    #plt.savefig(f"{file}/plots.png")
    plt.show()


### - - - - - - - - - - - - - - - - - - - - - - -


if __name__ == "__main__":
    plotplot()
