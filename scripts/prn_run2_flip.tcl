# Portable VMD scene: keep this file beside flip.xyz and frames.csv.
# Launch: vmd -e view_flip.tcl
# Atom labels below use the original one-based IDs.
namespace eval prnflip {
    variable directory [file dirname [file normalize [info script]]]
    variable rows {}
    variable molid
    variable hbonds [dict create]
}
set h [open [file join $prnflip::directory frames.csv] r]
gets $h header
while {[gets $h line] >= 0} {
    if {[string trim $line] ne ""} {lappend prnflip::rows [split $line ,]}
}
close $h
# Hydrogen bonds are computed using periodic source geometry before alignment.
set hbfile [file join $prnflip::directory hbonds.csv]
if {[file exists $hbfile]} {
    set h [open $hbfile r]
    gets $h header
    while {[gets $h line] >= 0} {
        if {[string trim $line] eq ""} {continue}
        set row [split $line ,]
        dict lappend prnflip::hbonds [lindex $row 0] $row
    }
    close $h
}
set prnflip::molid [mol new [file join $prnflip::directory flip.xyz] type xyz waitfor all autobonds off]
mol rename $prnflip::molid "PRN run-2 | anti to syn | 0–0.5 ps"
set all [atomselect $prnflip::molid all]
set connectivity [lrepeat 311 {}]
foreach pair {{0 1} {0 5} {0 6} {0 7} {1 2} {1 8} {1 9} {2 3} {2 4} {3 10}} {
    lassign $pair a b
    lset connectivity $a [concat [lindex $connectivity $a] $b]
    lset connectivity $b [concat [lindex $connectivity $b] $a]
}
set residues [lrepeat 11 1]
for {set o 11} {$o < 311} {incr o 3} {
    set h1 [expr {$o+1}]; set h2 [expr {$o+2}]
    lset connectivity $o [list $h1 $h2]
    lset connectivity $h1 [list $o]
    lset connectivity $h2 [list $o]
    set r [expr {2+($o-11)/3}]
    lappend residues $r $r $r
}
$all setbonds $connectivity
$all set resid $residues
$all delete
color Display Background white
display projection Orthographic
display depthcue off
axes location Off
display resize 1100 800
catch {display antialias on}
catch {display ambientocclusion on}
catch {display shadows on}
mol delrep 0 $prnflip::molid

proc prnflip::rep {selection style coloring material} {
    variable molid
    mol selection $selection
    mol representation {*}$style
    mol color {*}$coloring
    mol material $material
    mol addrep $molid
}
prnflip::rep "index 0 to 10" {CPK 0.65 0.18 24 24} {Name} Opaque
# Nearby complete waters; XYZ has no residue topology, so select OHH triplets explicitly.
prnflip::rep "none" {Licorice 0.065 12 12} {ColorID 6} Transparent
prnflip::rep "index 10" {VDW 0.38 32} {ColorID 11} Opaque

proc prnflip::update {args} {
    variable molid
    variable rows
    variable hbonds
    set frame [molinfo $molid get frame]
    if {$frame < 0 || $frame >= [llength $rows]} {return}
    set row [lindex $rows $frame]
    set selected [atomselect $molid "index >= 11 and name O and within 4.0 of (index 2 3 4 10)" frame $frame]
    set waters {}
    foreach o [$selected get index] {lappend waters $o [expr {$o+1}] [expr {$o+2}]}
    $selected delete
    if {[llength $waters]} {
        mol modselect 1 $molid "index [join $waters]"
    } else {mol modselect 1 $molid none}
    graphics $molid delete all
    set sol [atomselect $molid "index 0 to 10" frame $frame]
    set xyz [$sol get {x y z}]
    $sol delete
    graphics $molid color orange
    foreach pair {{4 2} {2 3} {3 10}} {
        lassign $pair a b
        graphics $molid line [lindex $xyz $a] [lindex $xyz $b] width 4 style dashed
    }
    graphics $molid color black
    foreach index {4 2 3 10} text {O2\ (5) CG\ (3) O1\ (4) H11\ (11)} {
        graphics $molid text [vecadd [lindex $xyz $index] {0.1 0.25 0.25}] $text size 1.1
    }
    set count 0
    if {[dict exists $hbonds $frame]} {
        foreach contact [dict get $hbonds $frame] {
            incr count
            set endpoint [lrange $contact 9 11]
            set kind [lindex $contact 5]
            if {$kind eq "solute"} {graphics $molid color cyan} else {graphics $molid color green}
            graphics $molid line [lindex $xyz 10] $endpoint width 5 style dashed
            graphics $molid sphere $endpoint radius 0.20 resolution 20
            set midpoint [vecscale 0.5 [vecadd [lindex $xyz 10] $endpoint]]
            graphics $molid text [vecadd $midpoint {0 0 0.25}] [format "O%d: H-A %.2f A, %.0f deg" [lindex $contact 4] [lindex $contact 7] [lindex $contact 8]] size 0.85
        }
    }
    graphics $molid color black
    graphics $molid text {-3.4 3.8 0} "PRN run-2: anti to syn" size 1.3
    graphics $molid text {-3.4 3.3 0} [format "t = %.2f ps     dihedral = %.1f deg" [lindex $row 1] [lindex $row 2]] size 1.1
    graphics $molid text {-3.4 2.8 0} "H11 hydrogen bonds: $count" size 1.0
    graphics $molid text {-3.4 -3.4 0} "Purple: H11 | Orange: torsion quartet | Gray: waters within 4 A" size 0.8
    graphics $molid text {-3.4 -3.8 0} "H11 H-bonds: green = solvent, cyan = solute" size 0.8
}
display resetview
# Fixed camera on the already aligned carboxyl group, not the full solvent box.
molinfo $prnflip::molid set center_matrix [transoffset {0 0 0}]
molinfo $prnflip::molid set scale_matrix [transscale 0.20]
rotate x by -20
rotate y by 20
trace add variable ::vmd_frame($prnflip::molid) write prnflip::update
animate goto 0
prnflip::update
animate style Loop
animate speed 0.15
puts "PRN run-2 flip: 0.13–0.19 ps (frames 13–19)."
puts "Play with VMD controls; Tk Console: animate forward / animate pause / animate goto 15"
puts "Solvent toggle: mol showrep $prnflip::molid 1 0 (off), or 1 (on)."
