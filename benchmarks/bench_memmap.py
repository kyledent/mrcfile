"""The out-of-core path: mmap open plus a statistics sweep."""
import _harness as h
import mrcfile

def main():
    path = h.fixture("vol_f32.mrc")
    h.warm(path)

    def mmap_open():
        with mrcfile.mmap(path) as m:
            return m.data[0, 0, 0]

    mrc = mrcfile.mmap(path, mode="r+")

    def mmap_stats():
        mrc.update_header_stats()

    h.emit("memmap", {
        "mmap_open": h.measure(mmap_open, repeats=10),
        "mmap_update_stats_128MiB": h.measure(mmap_stats),
    })
    mrc.close()

main()
