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
                elif highlight_stabilizers and is_stabilizer(tags[i], code):
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
    code = code_LMPZ
    file = f"saves_bf/{code.name}"
    wmax = code.n
    g, gc = 1, 1
    t = 0.5

    do_plot = True

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
    error_syndromes = get_syndrome(error_tags, code)
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
                                 sorted_syndrome_ind, sorted_syndrome_errors_ind, fuse_errors=False, show_error_tags=True, highlight_stabilizers=True, error_tag_lim=0, #figsize=(500,200),
                                 title=f"{code.name} (approx {code.n-wmax})  -  g = {g}  -  gc = {gc}  -  t = {t:.3e}  -  p = {round(( 1-IXYZ_proba[0] )*100,2)}%  -  p' = {round(( 1-fidelity )*100,2)}%")
        t1 = time.time(); print(f"\rplotting                     - {(t1-t0):.3g} s")

    print("\rentropies...", end=""); t0 = time.time()
    data_entropy = np.sum(entropy_form(error_proba))
    measure_entropy = np.sum(entropy_form(syndrome_proba))
    corrected_data_entropy = np.sum(entropy_form(corrected_error_proba))
    t1 = time.time(); print(f"\rentropies                    - {(t1-t0):.3g} s")

    T1 = time.time()
    print(f"\nTOTAL - {(T1-T0):.3g} s\n")

    if do_plot:
        plt.savefig("bigassplot.png")
        #plt.show()



### - - - - - - - - - - - - - - - - - - - - - - -



def multiple_brute_force():
    code = rotated_surface_code(5)
    file = f"saves_bf/{code.name}"
    wmax = 3
    g, gc = 1, 1
    times = np.logspace(-2.5,0,43) # 8 15 22 29 36 43 50 # -2,0.3 # -2.5,0

    do_plot = False
    save_name = f"_{wmax}"
    param = np.array((code, wmax, g, gc, times), dtype=object)
    np.save(f"{file}/param{save_name}", param)

    data_entropies           = np.array([])
    measure_entropies        = np.array([])
    corrected_data_entropies = np.array([])
    equivalence_entropies    = np.array([])
    error_probas             = np.array([])
    logical_probas           = np.empty((0,4))

    for i,t in enumerate(times):

        print(f"{i+1}/{len(times)}")

        IXYZ_proba = get_IXYZ_error_proba(t,g,gc)
        error_tags = get_tags(4, code.n, wmax)

        error_proba = get_proba(error_tags, IXYZ_proba, code.n)
        error_syndromes = get_syndrome(error_tags, code)
        syndrome_errors = find_syndrome_errors(error_syndromes)
        syndrome_tags = list(syndrome_errors.keys())
        syndrome_errors = list(syndrome_errors.values())

        syndrome_proba = get_syndrome_proba(syndrome_errors, error_proba)
        equivalence_proba = get_equivalence_proba(syndrome_errors, error_proba, error_tags, code)

        if do_plot:
            sorted_syndrome_ind, sorted_syndrome_errors_ind = sorted_syndrome_error_indices(syndrome_errors, error_proba, syndrome_proba)

        corrected_errors = apply_correction(syndrome_errors, error_proba, error_tags)
        corrected_error_tags = list(set(corrected_errors))
        corrected_error_ind = {e:i for i,e in enumerate(corrected_error_tags)}
        corrected_errors = [corrected_error_ind[corrected_errors[ei]] for ei in range(len(error_tags))]

        corrected_error_proba = get_new_proba(error_proba, corrected_errors, corrected_error_tags)
        logical_proba = get_logical_proba(corrected_error_proba, corrected_error_tags, code)

        if do_plot:
            plot_errors_to_syndromes(syndrome_errors, error_proba, syndrome_proba, error_tags, syndrome_tags, code, 
                                     sorted_syndrome_ind, sorted_syndrome_errors_ind, fuse_errors=True, show_error_tags=False, highlight_stabilizers=False,
                                     title=f"{code.name} (approx {code.n-wmax})  -  g = {g}  -  gc = {gc}  -  t = {t:.3e}  -  p = {round(( 1-IXYZ_proba[0] )*100,2)}%  -  p' = {round(( 1-logical_proba[0] )*100,2)}%")
            plt.savefig(f"{file}/{str(i).zfill(3)}.png")
            plt.close()
        
        data_entropy           = np.sum(entropy_form(error_proba))
        measure_entropy        = np.sum(entropy_form(syndrome_proba))
        corrected_data_entropy = np.sum(entropy_form(corrected_error_proba))
        equivalence_entropy    = np.sum(entropy_form(equivalence_proba))

        data_entropies           = np.append(data_entropies,           [data_entropy])
        measure_entropies        = np.append(measure_entropies,        [measure_entropy])
        corrected_data_entropies = np.append(corrected_data_entropies, [corrected_data_entropy])
        equivalence_entropies    = np.append(equivalence_entropies,    [equivalence_entropy])
        error_probas             = np.append(error_probas,             [1-IXYZ_proba[0]])
        logical_probas           = np.append(logical_probas,           [logical_proba], axis=0)

        data = np.empty(6, dtype=object)
        data[:] = (data_entropies, measure_entropies, corrected_data_entropies, equivalence_entropies, error_probas, logical_probas)
        np.save(f"{file}/data{save_name}", data)



### - - - - - - - - - - - - - - - - - - - - - - -



def bf_plotplot():
    code_name = "3RSC"
    file = f"saves_bf/{code_name}"
    extension = ""

    code, wmax, g, gc, times = np.load(f"{file}/param{extension}.npy", allow_pickle=True)
    data_entropies, measure_entropies, corrected_data_entropies, equivalence_entropies, error_probas, logical_probas = [np.array(L) for L in np.load(f"{file}/data{extension}.npy", allow_pickle=True)]
    fidelities = logical_probas[:,0]
    logical_corrected_entropies = np.sum(entropy_form(logical_probas), axis=1)
    efficiencies1 = (data_entropies - corrected_data_entropies)/measure_entropies
    efficiencies2 = (equivalence_entropies - logical_corrected_entropies)/measure_entropies
    breakeven = compute_breakeven(error_probas, 1-fidelities)

    #plt.rcParams["font.family"] = "Times New Roman"
    #plt.rcParams["font.size"] = 11
    fig, ax = plt.subplots(1,3,figsize=(12,4),dpi=300)

    ax[0].loglog(error_probas, data_entropies,              lw=2, c="crimson",        zorder=9,  label="data",           ls="--")
    ax[0].loglog(error_probas, corrected_data_entropies,    lw=2, c="deeppink",       zorder=10, label="corrected data", ls="--")
    ax[0].loglog(error_probas, equivalence_entropies,       lw=2, c="darkviolet",     zorder=12, label="equivalence")
    ax[0].loglog(error_probas, logical_corrected_entropies, lw=2, c="darkturquoise",  zorder=13, label="corrected logical")
    ax[0].loglog(error_probas, measure_entropies,           lw=2, c="mediumseagreen", zorder=11, label="measurement")
    ax[0].legend()
    ax[0].set_title("entropies")
    ax[0].grid(which='minor', linewidth=0.3, alpha=0.5)

    ax[1].loglog([0,error_probas[-1]**(1/code.n)], [0,error_probas[-1]**(1/code.n)], lw=1.5, ls="--", c="lightgrey", alpha=0.5)
    ax[1].loglog(error_probas, 1-fidelities, lw=2, c="mediumseagreen", zorder=9)
    ax[1].grid(which='minor', linewidth=0.3, alpha=0.5)
    ax[1].set_title(r"logical error probability $p'$" + ("" if breakeven is None else fr" ($p_\text{{BE}} = {round(breakeven*100,2)}\%$)"))
    ax[1].set_xlabel(r"physical error rate $p$")

    ax[2].loglog(error_probas, efficiencies1, lw=2, c="mediumseagreen", label="naive", ls="--")
    ax[2].loglog(error_probas, efficiencies2, lw=2, c="darkturquoise",  label="physical")
    ax[2].set_title("efficiency (-ies)")
    ax[2].grid(which='minor', linewidth=0.3, alpha=0.5)
    ax[2].legend()
    
    if not breakeven in (None, 0): 
        ax[0].scatter([breakeven], np.interp([breakeven],error_probas,data_entropies),              s=40, c="crimson",        zorder=17)
        ax[0].scatter([breakeven], np.interp([breakeven],error_probas,measure_entropies),           s=40, c="mediumseagreen", zorder=18)
        ax[0].scatter([breakeven], np.interp([breakeven],error_probas,corrected_data_entropies),    s=40, c="deeppink",       zorder=19)
        ax[0].scatter([breakeven], np.interp([breakeven],error_probas,equivalence_entropies),       s=40, c="darkviolet",     zorder=20)
        ax[0].scatter([breakeven], np.interp([breakeven],error_probas,logical_corrected_entropies), s=40, c="darkturquoise",  zorder=21)
        ax[1].scatter([breakeven], np.interp([breakeven],error_probas,1-fidelities),                s=40, c="mediumseagreen")
        ax[2].scatter([breakeven], np.interp([breakeven],error_probas,efficiencies1),               s=40, c="mediumseagreen")
        ax[2].scatter([breakeven], np.interp([breakeven],error_probas,efficiencies2),               s=40, c="darkturquoise")
    
    plt.tight_layout()
    plt.savefig(f"plotplot_{code_name}.png")
    #plt.show()



### - - - - - - - - - - - - - - - - - - - - - - -



def bf_compare_maxw():
    file = f"saves_bf/3RSC"
    extension = ""

    figE, axE = plt.subplots(3, 3, figsize=(15, 13), dpi=300)   # one entropy panel per wmax (9 -> 1)
    figL, axL = plt.subplots(1, 2, figsize=(11, 4.5), dpi=300)  # logical error proba + efficiency

    speeds = np.load(f"{file}/speed.npy")

    for w in range(9, 0, -1):
        code, wmax, g, gc, times = np.load(f"{file}/param_{w}.npy", allow_pickle=True)
        data_entropies, measure_entropies, corrected_data_entropies, equivalence_entropies, error_probas, logical_probas = [np.array(L) for L in np.load(f"{file}/data_{w}.npy", allow_pickle=True)]
        fidelities = logical_probas[:,0]
        logical_corrected_entropies = np.sum(entropy_form(logical_probas), axis=1)
        efficiencies1 = (data_entropies - corrected_data_entropies)/measure_entropies
        efficiencies2 = (equivalence_entropies - logical_corrected_entropies)/measure_entropies

        if wmax == 9:   # exact reference: draw the w=9 curves faintly on every other panel
            breakeven = compute_breakeven(error_probas, 1-fidelities)
            for i in range(1, 9):
                a = axE[i//3, i%3]
                a.loglog(error_probas, equivalence_entropies,       lw=3, c="black", alpha=0.2,  zorder=5)
                a.loglog(error_probas, logical_corrected_entropies, lw=3, c="black", alpha=0.1,  zorder=5)
                a.loglog(error_probas, measure_entropies,           lw=3, c="black", alpha=0.15, zorder=5)
            axL[0].loglog([error_probas[0], error_probas[-1]**(1/code.n)], [error_probas[0], error_probas[-1]**(1/code.n)], lw=1.5, ls="--", c="lightgrey", zorder=4, alpha=0.5)
            if breakeven not in (None, 0):
                axL[0].scatter([breakeven], np.interp([breakeven],error_probas,1-fidelities),  s=40, color=cm.PuBuGn(1.0), zorder=5)
                axL[1].scatter([breakeven], np.interp([breakeven],error_probas,efficiencies2), s=40, color=cm.PuBuGn(1.0), zorder=5)

        a = axE[(9-wmax)//3, (9-wmax)%3]
        if breakeven not in (None, 0):
            a.scatter([breakeven], np.interp([breakeven],error_probas,equivalence_entropies),       s=40, c="darkviolet",     zorder=5)
            a.scatter([breakeven], np.interp([breakeven],error_probas,measure_entropies),           s=40, c="mediumseagreen", zorder=5)
            a.scatter([breakeven], np.interp([breakeven],error_probas,logical_corrected_entropies), s=40, c="darkturquoise",  zorder=5)
        a.loglog(error_probas, equivalence_entropies,       lw=3, c="darkviolet",     zorder=10, label="equivalence")
        a.loglog(error_probas, logical_corrected_entropies, lw=3, c="darkturquoise",  zorder=10, label="corrected logical")
        a.loglog(error_probas, measure_entropies,           lw=3, c="mediumseagreen", zorder=10, label="measurement")
        a.set_title(fr"max weight $w_\text{{max}} = {wmax}$ - speed increase $\times {(speeds[0]/speeds[9-w]):.2g}$")
        a.grid(which='minor', linewidth=0.3, alpha=0.5)

        axL[0].loglog(error_probas, 1-fidelities, lw=1+wmax/3, zorder=20-wmax, c=cm.GnBu((wmax+1)/10), label=fr"$w_\text{{max}} = {wmax}$")
        axL[0].grid(which='minor', linewidth=0.3, alpha=0.5)

        axL[1].loglog(error_probas, efficiencies2, lw=1+wmax/3, zorder=20-wmax, c=cm.GnBu((wmax-1)/8))

    axL[0].grid(which='minor', linewidth=0.3, alpha=0.5)
    
    axL[1].set_title("efficiency")
    axL[1].grid(which='minor', linewidth=0.3, alpha=0.5)
    axL[1].set_ylim(0.6, 1.04)

    axL[0].set_title(r"logical error probability $p'$" + ("" if breakeven is None else fr" ($p_\text{{BE}} = {round(breakeven*100,2)}\%$)"))
    axE[0,0].legend()
    axE[2,1].set_xlabel(r"physical error rate $p$")
    axL[0].legend(title=r"$w_\text{max}$", fontsize=8, ncol=2, loc="lower right")

    figE.tight_layout()
    figL.tight_layout()
    figE.savefig("plot_approx_entropies.png")
    figL.savefig("plot_approx_logical_efficiency.png")
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



def monte_carlo_rsc(distance):
    code = rotated_surface_code(distance)
    g, gc = 1, 1
    times = np.logspace(-2, 0.3, 43) # 8 15 22 29 36 43 50
    shot0 = 100000
    shot = shot0

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
        
        shot = int(shot0*(1-(1-final[0]/shot)*1.3))
        finals.append(final)
        print(f"p = {(1-IXYZ_proba[0]):.3g} \t p' = {(1-final[0]/sum(final)):.3g} \t {final} \t shot = {shot}")
    np.save(f"saves_mc/data_{distance}", np.array([error_rates, finals], dtype=object))     


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


### - - - - - - - - - - - - - - - - - - - - - - -
### - - - - - - - - - - - - - - - - - - - - - - -


def plot_threshold_fit(distances=(3, 5, 7, 9), pmax=0.12, min_events=10):
    '''
    fit the threshold-theorem model with fit_threshold_mc() and overlay the fitted        \n
    curves on the monte-carlo data (log-log).
    '''
    A, B, p_th, perr = fit_threshold_mc(distances, pmax, min_events)
    colors = dict(zip((3, 5, 7, 9),
                      ["purple", "mediumslateblue", "cornflowerblue", "lightskyblue"]))

    fig, ax = plt.subplots(figsize=(7, 6))
    for d in distances:
        p, pL, N = load_mc_curve(d)
        c = colors.get(d, None)
        ax.loglog(p, pL, "o", ms=4, alpha=0.5, color=c, label=f"MC $d={d}$")
        pp = np.logspace(np.log10(p.min()), np.log10(min(pmax, p.max())), 200)
        ax.loglog(pp, A * (pp / p_th) ** (B * (d + 1) / 2), lw=2, color=c)

    ax.axvline(p_th, ls="--", c="grey", alpha=0.6)
    ax.set_xlabel("physical error rate $p$")
    ax.set_ylabel("logical error rate $p_L$")
    ax.set_title(rf"$p_L = A\,(p/p_{{th}})^{{B(d+1)/2}}$   "
                 rf"($A={A:.2g}$, $B={B:.2g}$, $p_{{th}}={p_th:.3g}$)")
    ax.grid(which="minor", linewidth=0.3, alpha=0.5)
    ax.legend()
    plt.tight_layout()
    plt.show()
    return A, B, p_th, perr


### - - - - - - - - - - - - - - - - - - - - - - -
### - - - - - - - - - - - - - - - - - - - - - - -


def plot_rsc_compare():
    fig, (axL, axE) = plt.subplots(1, 2, figsize=(11, 6))#, dpi=300)
    
    rsc3_9_p, rsc3_9_pL, rsc3_9_eff = load_bf_curve("3RSC")      
    rsc3_3_p, rsc3_3_pL, rsc3_3_eff = load_bf_curve("3RSC", "_3")
    rsc5_5_p, rsc5_5_pL, rsc5_5_eff = load_bf_curve("5RSC", "_5")
    rsc5_4_p, rsc5_4_pL, rsc5_4_eff = load_bf_curve("5RSC", "_4")
    rsc5_3_p, rsc5_3_pL, rsc5_3_eff = load_bf_curve("5RSC", "_3")

    threshold_bf = compute_threshold(rsc3_9_p, rsc3_9_pL, rsc5_5_p, rsc5_5_pL)

    mc_dists  = (3, 5, 7, 9)
    mc_colors = ("purple", "mediumslateblue", "cornflowerblue", "lightskyblue")
    mc = {d: load_mc_curve(d) for d in mc_dists}

    threshold_mc = np.mean([
        compute_threshold(mc[da][0], mc[da][1], mc[db][0], mc[db][1])
        for da, db in zip(mc_dists[:-1], mc_dists[1:])
    ])
    threshold_mc_y = np.mean([
        np.interp(threshold_mc, p, pL) for p, pL, _ in mc.values()
    ])

    for zorder, (d, color) in zip(range(16, 12, -1), zip(mc_dists, mc_colors)):
        p, pL, _ = mc[d]
        axL.loglog(p, pL, c=color, lw=7, zorder=zorder, label=f"MC $d={d}$", alpha=0.5)
        if d == mc_dists[0]:
            xlim, ylim = axL.get_xlim(), axL.get_ylim()
            lo, hi = min(xlim[0], ylim[0]), max(xlim[1], ylim[1])
    axL.scatter([threshold_mc], [threshold_mc_y], s=500, color="violet", alpha=0.7, zorder=12)

    axL.loglog(rsc3_9_p, rsc3_9_pL, lw=2, c="purple",          zorder=20, label=r"exact $d=3$", ls="--")
    axL.loglog(rsc3_3_p, rsc3_3_pL, lw=2, c="purple",          zorder=21, label=r"$3$-approx $d=3$")
    axL.loglog(rsc5_5_p, rsc5_5_pL, lw=2, c="mediumslateblue", zorder=22, label=r"$5$-approx $d=5$")
    axL.loglog(rsc5_4_p, rsc5_4_pL, lw=2, c="mediumslateblue", zorder=22, label=r"$4$-approx $d=5$", alpha=0.7, ls="-.")
    axL.loglog(rsc5_3_p, rsc5_3_pL, lw=2, c="mediumslateblue", zorder=22, label=r"$4$-approx $d=5$", alpha=0.5, ls=":") 
    axL.scatter([threshold_bf], np.interp([threshold_bf], rsc3_3_p, rsc3_3_pL), s=50, color="purple", zorder=21)

    axL.plot([lo, hi], [lo, hi], lw=1.5, ls="--", c="lightgrey", zorder=5, alpha=0.5)
    axL.set_xlim(0.05,0.3); axL.set_ylim(0.03,0.9) #axL.set_xlim(xlim); axL.set_ylim(ylim)

    axE.loglog(rsc3_9_p, rsc3_9_eff, lw=2, c="purple",          zorder=20, label=r"exact $d=3$", ls="--")
    axE.loglog(rsc3_3_p, rsc3_3_eff, lw=2, c="purple",          zorder=21, label=r"$3$-approx $d=3$")
    axE.loglog(rsc5_5_p, rsc5_5_eff, lw=2, c="mediumslateblue", zorder=22, label=r"$5$-approx $d=5$")
    axE.loglog(rsc5_4_p, rsc5_4_eff, lw=2, c="mediumslateblue", zorder=22, label=r"$4$-approx $d=5$", alpha=0.7, ls="-.")
    axE.loglog(rsc5_3_p, rsc5_3_eff, lw=2, c="mediumslateblue", zorder=22, label=r"$3$-approx $d=5$", alpha=0.5, ls=":") 

    axL.set_title(r"logical error probability $p'$")
    axE.set_title("efficiency")
    axL.set_xlabel(r"physical error rate $p$")
    axE.set_xlabel(r"physical error rate $p$")
    axL.grid(which='minor', linewidth=0.3, alpha=0.5)
    axE.grid(which='minor', linewidth=0.3, alpha=0.5)
    axE.set_ylim(0.9, None)
    axL.legend()

    plt.tight_layout()
    plt.show()



### - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -



if __name__ == "__main__":
    plot_rsc_compare()
