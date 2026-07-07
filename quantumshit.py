print("\rbip...", end="")
from QEC import *
print("\rbip boup...\n")



### - - - BRUTE FORCE - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
### - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -



def plot_errors_to_syndromes(syndrome_errors, error_proba, syndrome_proba, error_tags, syndrome_tags, code, syndrome_ind=None, syndrome_errors_ind=None,
                             fuse_errors=False, colormap=cm.coolwarm, title=None, figsize=None, show_error_tags=True, error_tag_lim=None, highlight_stabilizers=False):
    if syndrome_ind is None or syndrome_errors_ind is None: syndrome_ind, syndrome_errors_ind = sorted_syndrome_error_indices(syndrome_errors, error_proba, syndrome_proba)
    if figsize is None: figsize = (max(20,len(syndrome_tags))/2, max(9,9 + np.log(len(syndrome_tags)/64)))
    if error_tag_lim is None: error_tag_lim = 0.02*figsize[1]/9
    fig,ax = plt.subplots(figsize=figsize)

    shifts = np.arange(code.n) * 2
    weights_all = ((np.asarray(error_tags)[:, None] >> shifts) & 3 != 0).sum(axis=1)

    all_verts = []
    all_colors = []
    for s_pos,si in enumerate(syndrome_ind):
        errors_ind = syndrome_errors_ind[si]
        w0 = None
        heights = [0]
        colors = []
        tags = []
        for i in range(len(errors_ind)):
            weight = weights_all[errors_ind[i]]
            if fuse_errors and weight == w0:
                heights[-1] += error_proba[errors_ind[i]]
                tags[-1] = -1
            else:
                heights.append(heights[-1] + error_proba[errors_ind[i]])
                colors.append(colormap( weight/(code.n+0.25) + (0 if fuse_errors else np.random.random()*0.25/(code.n+0.25)) ))
                tags.append(error_tags[errors_ind[i]])
                w0 = weight
        x0 = s_pos - 0.4
        for i in range(len(heights)-1):
            h = heights[i+1]-heights[i]
            y0 = heights[i]
            all_verts.append([(x0, y0), (x0+0.8, y0), (x0+0.8, y0+h), (x0, y0+h)])
            all_colors.append(colors[i])
            if show_error_tags and tags[i] != -1 and h/syndrome_proba[syndrome_ind[0]] > error_tag_lim:
                if highlight_stabilizers and tags[i] in code.generators:
                    ax.text(s_pos, heights[i]+h/2, letterify(tags[i],code.n), ha="center", va="center", color="black", fontfamily="monospace", fontsize=6, fontweight="bold")
                elif highlight_stabilizers and code.is_stabilizer(tags[i]):
                    ax.text(s_pos, heights[i]+h/2, letterify(tags[i],code.n), ha="center", va="center", color="black", fontfamily="monospace", fontsize=6)
                else :
                    ax.text(s_pos, heights[i]+h/2, letterify(tags[i],code.n), ha="center", va="center", color="white", fontfamily="monospace", fontsize=6)
    ax.add_collection(PolyCollection(all_verts, facecolors=all_colors, edgecolors="none"), autolim=False)
    ax.set_xlim(-0.5, len(syndrome_tags)-0.5)
    ax.set_ylim(0, syndrome_proba[syndrome_ind[0]])
    ax.set_xticks(range(len(syndrome_tags)))
    ax.set_xticklabels([binrepr(s,code.m) for s in syndrome_tags], rotation=90)
    if not title is None: fig.suptitle(title)
    plt.tight_layout()



### - - - - - - - - - - - - - - - - - - - - - - -



def single_brute_force():
    code = rotated_surface_code(3)
    file = f"saves_bf/{code.name}"
    wmax = 6
    g, gc = 1, 1
    t = 1

    do_plot = False

    T0 = time.time()

    print("\rIXYZ_proba...", end=""); t0 = time.time()
    IXYZ_proba = get_IXYZ_error_proba(t,g,gc)
    t1 = time.time(); print(f"\rIXYZ_proba                   - {(t1-t0):.3g} s")

    print("\rerror_tags...", end=""); t0 = time.time()
    error_tags = get_tags(4, code.n, wmax)
    t1 = time.time(); print(f"\rerror_tags                   - {(t1-t0):.3g} s")

    print("\rerror_proba...", end=""); t0 = time.time()
    error_proba = get_proba(error_tags, IXYZ_proba, code.n)
    t1 = time.time(); print(f"\rerror_proba                  - {(t1-t0):.3g} s")

    print("\rerror_syndrome...", end=""); t0 = time.time()
    error_syndromes = get_syndrome(error_tags, code.code, code.n)
    t1 = time.time(); print(f"\rerror_syndrome               - {(t1-t0):.3g} s")

    print("\rsyndrome_errors...", end=""); t0 = time.time()
    syndrome_errors = find_syndrome_errors(error_syndromes)
    t1 = time.time(); print(f"\rsyndrome_errors              - {(t1-t0):.3g} s")

    print("\rsyndrome_tags/errors...", end=""); t0 = time.time()
    syndrome_tags = list(syndrome_errors.keys())
    syndrome_errors = list(syndrome_errors.values())
    t1 = time.time(); print(f"\rsyndrome_tags/errors         - {(t1-t0):.3g} s")

    print("\rsyndrome_proba...", end=""); t0 = time.time()
    syndrome_proba = get_syndrome_proba(syndrome_errors, error_proba) 
    t1 = time.time(); print(f"\rsyndrome_proba               - {(t1-t0):.3g} s")

    if do_plot:
        print("\rsorted_syndrome_(errors)_ind...", end=""); t0 = time.time()
        sorted_syndrome_ind, sorted_syndrome_errors_ind = sorted_syndrome_error_indices(syndrome_errors, error_proba, syndrome_proba)
        t1 = time.time(); print(f"\rsorted_syndrome_(errors)_ind - {(t1-t0):.3g} s")

    print("\rcorrected_errors...", end=""); t0 = time.time()
    corrected_errors = apply_correction(syndrome_errors, error_proba, error_tags)
    t1 = time.time(); print(f"\rcorrected_errors             - {(t1-t0):.3g} s")

    print("\rcorrected_error_(tags/ind)...", end=""); t0 = time.time()
    corrected_error_tags = list(set(corrected_errors))
    corrected_error_ind = {e:i for i,e in enumerate(corrected_error_tags)}
    corrected_errors = [corrected_error_ind[corrected_errors[ei]] for ei in range(len(error_tags))]
    t1 = time.time(); print(f"\rcorrected_error_(tags/ind)   - {(t1-t0):.3g} s")

    print("\rcorrected_error_proba...", end=""); t0 = time.time()
    corrected_error_proba = get_new_proba(error_proba, corrected_errors, corrected_error_tags)
    t1 = time.time(); print(f"\rcorrected_error_proba        - {(t1-t0):.3g} s")

    print("\rfidelity...", end=""); t0 = time.time()
    fidelity = get_fidelity(corrected_error_proba, corrected_error_tags, code)
    t1 = time.time(); print(f"\rfidelity                     - {(t1-t0):.3g} s")

    if do_plot:
        print("\rplotting...", end=""); t0 = time.time()
        plot_errors_to_syndromes(syndrome_errors, error_proba, syndrome_proba, error_tags, syndrome_tags, code, 
                                 sorted_syndrome_ind, sorted_syndrome_errors_ind, fuse_errors=True, show_error_tags=False, highlight_stabilizers=False, error_tag_lim=0, #figsize=(500,200),
                                 title=f"{code.name} (approx {code.n-wmax})  -  g = {g}  -  gc = {gc}  -  t = {t:.3e}  -  p = {round(( 1-IXYZ_proba[0] )*100,2)}%  -  p' = {round(( 1-fidelity )*100,2)}%")
        t1 = time.time(); print(f"\rplotting                     - {(t1-t0):.3g} s")

    print("\rentropies...", end=""); t0 = time.time()
    data_entropy = np.sum(entropy_form(error_proba))
    measure_entropy = np.sum(entropy_form(syndrome_proba))
    corrected_data_entropy = np.sum(entropy_form(corrected_error_proba))
    t1 = time.time(); print(f"\rentropies                    - {(t1-t0):.3g} s")

    T1 = time.time()
    print(f"\nTOTAL - {(T1-T0):.3g} s\n")

    plt.savefig("bigassplot.png")
    #plt.show()



### - - - - - - - - - - - - - - - - - - - - - - -



def multiple_brute_force():
    code = rotated_surface_code(5)
    file = f"saves_bf/{code.name}"
    wmax = 6
    g, gc = 1, 1
    times = np.logspace(-2,0.3,15) # 8 15 22 29 36 43 50

    do_plot = False
    save_name = f"_{wmax}"
    param = np.array((code, wmax, g, gc, times), dtype=object)
    np.save(f"{file}/param{save_name}", param)

    data_entropies           = np.array([])
    measure_entropies        = np.array([])
    corrected_data_entropies = np.array([])
    error_probas             = np.array([])
    fidelities               = np.array([])

    for i,t in enumerate(times):

        IXYZ_proba = get_IXYZ_error_proba(t,g,gc)
        error_tags = get_tags(4, code.n, wmax)

        error_proba = get_proba(error_tags, IXYZ_proba, code.n)
        error_syndromes = get_syndrome(error_tags, code.code, code.n)
        syndrome_errors = find_syndrome_errors(error_syndromes)
        syndrome_tags = list(syndrome_errors.keys())
        syndrome_errors = list(syndrome_errors.values())

        syndrome_proba = get_syndrome_proba(syndrome_errors, error_proba) 

        if do_plot:
            sorted_syndrome_ind, sorted_syndrome_errors_ind = sorted_syndrome_error_indices(syndrome_errors, error_proba, syndrome_proba)

        corrected_errors = apply_correction(syndrome_errors, error_proba, error_tags)
        corrected_error_tags = list(set(corrected_errors))
        corrected_error_ind = {e:i for i,e in enumerate(corrected_error_tags)}
        corrected_errors = [corrected_error_ind[corrected_errors[ei]] for ei in range(len(error_tags))]

        corrected_error_proba = get_new_proba(error_proba, corrected_errors, corrected_error_tags)
        fidelity = get_fidelity(corrected_error_proba, corrected_error_tags, code)

        if do_plot:
            plot_errors_to_syndromes(syndrome_errors, error_proba, syndrome_proba, error_tags, syndrome_tags, code, 
                                     sorted_syndrome_ind, sorted_syndrome_errors_ind, fuse_errors=True, show_error_tags=False, highlight_stabilizers=False,
                                     title=f"{code.name} (approx {code.n-wmax})  -  g = {g}  -  gc = {gc}  -  t = {t:.3e}  -  p = {round(( 1-IXYZ_proba[0] )*100,2)}%  -  p' = {round(( 1-fidelity )*100,2)}%")
            plt.savefig(f"{file}/{str(i).zfill(3)}.png")
            plt.close()
        
        data_entropy = np.sum(entropy_form(error_proba))
        measure_entropy = np.sum(entropy_form(syndrome_proba))
        corrected_data_entropy = np.sum(entropy_form(corrected_error_proba))

        data_entropies           = np.append(data_entropies,           [data_entropy])
        measure_entropies        = np.append(measure_entropies,        [measure_entropy])
        corrected_data_entropies = np.append(corrected_data_entropies, [corrected_data_entropy])
        error_probas             = np.append(error_probas,             [1-IXYZ_proba[0]])
        fidelities               = np.append(fidelities,               [fidelity])
        np.save(f"{file}/data{save_name}", (data_entropies, measure_entropies, corrected_data_entropies, error_probas, fidelities))



### - - - - - - - - - - - - - - - - - - - - - - -



def bf_plotplot():
    file = "saves_bf/LMPZ"
    extension = ""

    code, wmax, g, gc, times = np.load(f"{file}/param{extension}.npy", allow_pickle=True)
    data_entropies, measure_entropies, corrected_data_entropies, error_probas, fidelities = [np.array(L) for L in np.load(f"{file}/data{extension}.npy")]
    #IXYZ_probas = np.array([get_IXYZ_error_proba(t,g,gc) for t in times])
    #single_entropies = np.sum(entropy_form(IXYZ_probas), axis=1)
    #logipX, logipY, logipZ = [ (1-fidelities)/3 ]*3
    #corrected_single_entropies = - fidelities*np.log(fidelities) - logipX*np.log(logipX) - logipY*np.log(logipY) - logipZ*np.log(logipZ)
    efficiencies = (data_entropies - corrected_data_entropies)/measure_entropies
    #efficiencies2 = (single_entropies - corrected_single_entropies)/measure_entropies
    threashold = compute_threashold(error_probas, 1-fidelities)

    #plt.rcParams["font.family"] = "Times New Roman"
    #plt.rcParams["font.size"] = 11
    fig, ax = plt.subplots(1,3,figsize=(12,4),dpi=300)

    #ax[0].loglog(error_probas, single_entropies,           lw=2, c="royalblue",      zorder=7,  label="logical")
    #ax[0].loglog(error_probas, corrected_single_entropies, lw=2, c="darkturquoise",  zorder=8,  label="corrected logical")
    ax[0].loglog(error_probas, data_entropies,             lw=2, c="crimson",        zorder=9,  label="data")
    ax[0].loglog(error_probas, corrected_data_entropies,   lw=2, c="deeppink",       zorder=10, label="corrected data")
    ax[0].loglog(error_probas, measure_entropies,          lw=2, c="mediumseagreen", zorder=11, label="measurement")
    ax[0].legend()
    ax[0].set_title("entropy")
    ax[0].grid(which='minor', linewidth=0.3, alpha=0.5)

    ax[1].loglog([0,error_probas[-1]**(1/code.n)], [0,error_probas[-1]**(1/code.n)], lw=1.5, ls="--", c="lightgrey", alpha=0.5)
    ax[1].loglog(error_probas, 1-fidelities, lw=2, c="mediumseagreen")
    ax[1].grid(which='minor', linewidth=0.3, alpha=0.5)
    ax[1].set_title("logical error probability" + ("" if threashold is None else f" (p_th = {round(threashold*100,2)}%)"))

    ax[2].loglog(error_probas, efficiencies, lw=2, c="mediumseagreen")
    #ax[2].loglog(error_probas, efficiencies2, lw=2, c="royalblue")
    ax[2].set_title("efficiency")
    ax[2].grid(which='minor', linewidth=0.3, alpha=0.5)
    
    if not threashold in (None, 0): 
        #ax[0].scatter([threashold], np.interp([threashold],error_probas,single_entropies),           s=40, c="royalblue",      zorder=15)
        #ax[0].scatter([threashold], np.interp([threashold],error_probas,corrected_single_entropies), s=40, c="darkturquoise",  zorder=16)
        ax[0].scatter([threashold], np.interp([threashold],error_probas,data_entropies),             s=40, c="crimson",        zorder=17)
        ax[0].scatter([threashold], np.interp([threashold],error_probas,measure_entropies),          s=40, c="mediumseagreen", zorder=18)
        ax[0].scatter([threashold], np.interp([threashold],error_probas,corrected_data_entropies),   s=40, c="deeppink",       zorder=19)
        ax[1].scatter([threashold], np.interp([threashold],error_probas,1-fidelities),               s=40, c="mediumseagreen")
        ax[2].scatter([threashold], np.interp([threashold],error_probas,efficiencies),               s=40, c="mediumseagreen")
        #ax[2].scatter([threashold], np.interp([threashold],error_probas,efficiencies2),              s=40, c="royalblue")
    
    plt.tight_layout()
    plt.savefig("plotplot.png") #plt.savefig(f"{file}/{code.name}_plots{extension}.png")
    #plt.show()



### - - - - - - - - - - - - - - - - - - - - - - -



def bf_plot_compare1():
    #plt.rcParams["font.family"] = "Times New Roman"
    #plt.rcParams["font.size"] = 11
    fig, ax = plt.subplots(1,3,figsize=(12,4),dpi=300)

    for wmax in range(5,1,-1):
        file = "saves_bf/LMPZ"
        print(wmax)

        code, wmax, g, gc, times = np.load(f"{file}/param_{wmax}.npy", allow_pickle=True)
        data_entropies, measure_entropies, corrected_data_entropies, error_probas, fidelities = [np.array(L[10:]) for L in np.load(f"{file}/data_{wmax}.npy")]
        efficiencies = (data_entropies - corrected_data_entropies)/measure_entropies
        threashold = compute_threashold(error_probas, 1-fidelities)

        ax[0].loglog(error_probas, data_entropies, lw=.5+wmax/2, c="crimson", label="data", alpha=(wmax-1)/4)
        ax[0].loglog(error_probas, corrected_data_entropies, lw=.5+wmax/2, c="deeppink", label="corrected data", alpha=(wmax-1)/4)
        ax[0].loglog(error_probas, measure_entropies, lw=.5+wmax/2, c="mediumseagreen", label="measurement", alpha=(wmax-1)/4)
        ax[0].set_title("entropy")
        ax[0].grid(which='minor', linewidth=0.3, alpha=0.5)
        ax[0].set_ylim(0.04,10)

        ax[1].loglog(error_probas, 1-fidelities, lw=.5+wmax/2, c="mediumseagreen", alpha=(wmax-1)/4)
        ax[1].grid(which='minor', linewidth=0.3, alpha=0.5)
        ax[1].set_title("logical error probability" + ("" if threashold is None else f" (p_th = {round(threashold*100,2)}%)"))

        ax[2].loglog(error_probas, efficiencies, lw=.5+wmax/2, c="mediumseagreen", alpha=(wmax-1)/4)
        ax[2].set_title("efficiency")
        ax[2].grid(which='minor', linewidth=0.3, alpha=0.5)
        ax[2].set_ylim(0.5,1.04)

        if not threashold in (None, 0) and wmax == 5: 
            ax[1].loglog([0,error_probas[-1]**(1/code.n)], [0,error_probas[-1]**(1/code.n)], lw=1.5, ls="--", c="lightgrey", alpha=0.5)
            ax[0].scatter([threashold], np.interp([threashold],error_probas,data_entropies),         alpha=wmax/5, s=40, c="crimson")
            ax[0].scatter([threashold], np.interp([threashold],error_probas,measure_entropies),       alpha=wmax/5, s=40, c="mediumseagreen")
            ax[0].scatter([threashold], np.interp([threashold],error_probas,corrected_data_entropies), alpha=wmax/5, s=40, c="deeppink")
            ax[1].scatter([threashold], np.interp([threashold],error_probas,1-fidelities),              alpha=wmax/5, s=40, c="mediumseagreen")
            ax[2].scatter([threashold], np.interp([threashold],error_probas,efficiencies),               alpha=wmax/5, s=40, c="mediumseagreen")

    plt.tight_layout()
    plt.savefig(f"{file}/{code.name}_plots_approx.png")



def bf_plot_compare2():
    fig, ax = plt.subplots(2,4,figsize=(18,8),dpi=300)
    file = "saves_bf/3RSC"

    for wmax in range(9,3,-1):
        print(wmax)
        code, wmax, g, gc, times = np.load(f"{file}/param_{wmax}.npy", allow_pickle=True)
        data_entropies, measure_entropies, corrected_data_entropies, error_probas, fidelities = [np.array(L) for L in np.load(f"{file}/data_{wmax}.npy")]
        efficiencies = (data_entropies - corrected_data_entropies)/measure_entropies
        threashold = compute_threashold(error_probas, 1-fidelities)

        if wmax == 9:
            for i in range(1,6):
                ax[i//3,i%3].loglog(error_probas, data_entropies, lw=3, c="black", alpha=0.35, zorder=5)
                ax[i//3,i%3].loglog(error_probas, corrected_data_entropies, lw=3, c="black", alpha=0.3, zorder=5)
                ax[i//3,i%3].loglog(error_probas, measure_entropies, lw=3, c="black", alpha=0.2, zorder=5)
                if not threashold in (None, 0): 
                    ax[i//3,i%3].scatter([threashold], np.interp([threashold],error_probas,data_entropies),         s=40, c="crimson", zorder=5)
                    ax[i//3,i%3].scatter([threashold], np.interp([threashold],error_probas,measure_entropies),       s=40, c="mediumseagreen", zorder=5)
                    ax[i//3,i%3].scatter([threashold], np.interp([threashold],error_probas,corrected_data_entropies), s=40, c="deeppink", zorder=5)
            ax[0,3].loglog([0,error_probas[-1]**(1/code.n)], [0,error_probas[-1]**(1/code.n)], lw=1.5, ls="--", c="lightgrey", zorder=4, alpha=0.5)
            ax[0,3].scatter([threashold], np.interp([threashold],error_probas,1-fidelities), s=40, c=cm.PuBuGn(1.0), zorder=5)
            ax[1,3].scatter([threashold], np.interp([threashold],error_probas,efficiencies), s=40, c=cm.PuBuGn(1.0), zorder=5)

        ax[(9-wmax)//3,(9-wmax)%3].loglog(error_probas, data_entropies, lw=3, c="crimson", zorder=10, label="data")
        ax[(9-wmax)//3,(9-wmax)%3].loglog(error_probas, corrected_data_entropies, lw=3, c="deeppink", zorder=10, label="corrected data")
        ax[(9-wmax)//3,(9-wmax)%3].loglog(error_probas, measure_entropies, lw=3, c="mediumseagreen", zorder=10, label="measurement")
        ax[(9-wmax)//3,(9-wmax)%3].set_title(f"entropy (w_max = {wmax})")
        ax[(9-wmax)//3,(9-wmax)%3].grid(which='minor', linewidth=0.3, alpha=0.5)
        #ax[0].set_ylim(0.04,10)

        ax[0,3].loglog(error_probas, 1-fidelities, lw=wmax/3, zorder=20-wmax, c=cm.PuBuGn((wmax-1)/8))
        ax[0,3].grid(which='minor', linewidth=0.3, alpha=0.5)
        ax[0,3].set_title("logical error probability" + ("" if threashold is None else f" (p_th = {round(threashold*100,2)}%)"))

        ax[1,3].loglog(error_probas, efficiencies, lw=wmax/3, zorder=20-wmax, c=cm.PuBuGn((wmax-1)/8))
        ax[1,3].set_title("efficiency")
        ax[1,3].grid(which='minor', linewidth=0.3, alpha=0.5)
        ax[1,3].set_ylim(0.6,1.04)

    plt.tight_layout()
    plt.savefig(f"{file}/{code.name}_plots_approx.png")
    #plt.show()



### - - - MONTE CARLO - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
### - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -



def monte_carlo():
    code = rotated_surface_code(5)
    g, gc = 1, 1
    times = np.logspace(-2, 0.3, 50) # 8 15 22 29 36 43 50
    shot = 1000
    max_weight = 3

    error_rates = []
    finals = []

    decode = MLDecoder(code, get_IXYZ_error_proba(times[0],g,gc), max_weight)

    for it,t in enumerate(times):

        IXYZ_proba = get_IXYZ_error_proba(t,g,gc)
        decode.set_noise(IXYZ_proba)

        error_rates.append(1-IXYZ_proba[0])
        final = [0,0,0,0]

        for _ in range(shot):
            err = random_tag(code.n, IXYZ_proba)
            if err == 0:
                final[0] += 1
                continue
            syn = int(get_syndrome(err, code))
            cor = decode(syn)
            res = err ^ cor
            log = logical(res, code)
            final[log] += 1
        
        finals.append(final)
        print(f"p = {(1-IXYZ_proba[0]):.3g} \t p' = {(1-final[0]/sum(final)):.3g} \t {final}")
    np.save("data", np.array([error_rates, finals], dtype=object))     



def monte_carlo_rsc():
    distance = 9
    code = rotated_surface_code(distance)
    g, gc = 1, 1
    times = np.logspace(-2, 0.3, 50) # 8 15 22 29 36 43 50
    shot = 10000

    error_rates = []
    finals = []

    decode = MWPMDecoder(code, get_IXYZ_error_proba(times[0],g,gc))

    for it,t in enumerate(times):

        IXYZ_proba = get_IXYZ_error_proba(t,g,gc)
        decode.set_noise(IXYZ_proba)

        error_rates.append(1-IXYZ_proba[0])
        final = [0,0,0,0]

        for _ in range(shot):
            err = random_tag(code.n, IXYZ_proba)
            if err == 0:
                final[0] += 1
                continue
            syn = int(get_syndrome(err, code))
            cor = decode(syn)
            res = err ^ cor
            log = logical(res, code)
            final[log] += 1
        
        finals.append(final)
        print(f"p = {(1-IXYZ_proba[0]):.3g} \t p' = {(1-final[0]/sum(final)):.3g} \t {final}")
    np.save(f"saves_mc/RSC/data_{distance}", np.array([error_rates, finals], dtype=object))     


### - - - - - - - - - - - - - - - - - - - - - - -



def plot_ml():
    error_rates_3, finals_3 = np.load("saves_mc/rsc/data_3.npy", allow_pickle=True)
    error_rates_5, finals_5 = np.load("saves_mc/rsc/data_5.npy", allow_pickle=True)
    error_rates_7, finals_7 = np.load("saves_mc/rsc/data_7.npy", allow_pickle=True)
    error_rates_9, finals_9 = np.load("saves_mc/rsc/data_9.npy", allow_pickle=True)
    finals_3 = np.array([f for f in finals_3])
    finals_5 = np.array([f for f in finals_5])
    finals_7 = np.array([f for f in finals_7])
    finals_9 = np.array([f for f in finals_9])
    shot = np.sum((finals_3[0]))

    fig,ax = plt.subplots()

    ax.loglog(error_rates_3, 1-finals_3[:,0]/shot, c="red",    lw=5, zorder=13, label="d=3")
    ax.loglog(error_rates_5, 1-finals_5[:,0]/shot, c="purple", lw=5, zorder=14, label="d=5")
    ax.loglog(error_rates_7, 1-finals_7[:,0]/shot, c="blue",   lw=5, zorder=15, label="d=7")
    ax.loglog(error_rates_9, 1-finals_9[:,0]/shot, c="green",  lw=5, zorder=16, label="d=9")

    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    lo, hi = min(xlim[0], ylim[0]), max(xlim[1], ylim[1])

    #ax.loglog(error_rates,   finals[:,1]/shot, c="royalblue",       zorder=10)
    #ax.loglog(error_rates,   finals[:,2]/shot, c="red",             zorder=11)
    #ax.loglog(error_rates,   finals[:,3]/shot, c="mediumvioletred", zorder=12)

    ax.plot([lo, hi], [lo, hi], lw=1.5, ls="--", c="lightgrey", zorder=5, alpha=0.5)
    ax.set_xlim(xlim); ax.set_ylim(ylim)

    ax.grid(which='minor', linewidth=0.3, alpha=0.5)
    ax.legend()

    plt.tight_layout()
    plt.show()



### - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -



if __name__ == "__main__":
    monte_carlo_rsc()
    #plot_ml()
