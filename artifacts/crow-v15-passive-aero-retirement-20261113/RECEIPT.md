# Retirement receipt

After manifest commit `abd9d03ee1097358bebba8c225576f2dab62810e`, the four listed files were rechecked for their recorded inode, size, link count, SHA-256 and absence of open handles, then unlinked. The remote host reported 10,061,000 available 4096-byte blocks immediately before and 10,069,023 after; concurrent activity means this delta cannot be attributed solely to the deletion. The corresponding evidence remains independently restorable from the archived gzip files. These were the only paths retired by this operation.
