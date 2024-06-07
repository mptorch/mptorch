## Computer Systems


| *Name*             | *CPU*     | *Cores* | *GHz*             | *Threads* | *CPUmark*      | *RAM* | *Disk1*      | *Disk2*   | *GPU*            |
|:-------------------|----------:|:-------:|:-----------------:|----------:|---------------:|------:|-------------:|----------:|------------------|
| *mxp*              | i9-13900K | 8P + 12E| 3.0/5.8 + 2.2/4.3 | 16+16     | 4,632 (59,298) | 192GB | ssd:2TB      | ssd:4TB(/)| RTX 4090 24GB    |
| *rvv*              | i9-13900K | 8P + 12E| 3.0/5.8 + 2.2/4.3 | 16+16     | 4,632 (59,298) | 192GB | ssd:2TB      | ssd:4TB(/)| RTX 4090 24GB    |
| *gomeisa*          | i7-12700  | 8P + 4E | 2.1/4.9 + 1.6/3.6 | 16+4      | 3,915 (30,809) | 128GB | ssd:2TB(/)   | hdd:none  | Titan V 24GB     |
|                    |           |         |                   |           |                |       |              |           |                  |
| *mmx* (hipster)    | i7-4790   | 4       | 3.6/4.0           |    8      | 2,233  (7,268) | 32GB  | ssd:1TB(/)   | hdd:1TB   | none             |
| *orthanc*          | i7-10710U | 6       | 1.1/4.7           |   12      | 2,351  (9,607) | 64GB  | ssd:500GB(/) | hdd:none  | none             |
| *ave* (silverstone)| i7-5930K  | 6       | 3.5/3.7           |   12      | 2,050 (10,328) | 32GB  | ssd:1TB(/)   | hdd:400GB | Titan V 24GB     |
|                    |           |         |                   |           |                |       |              |           |                  |
| *sve*              | E5-1225v2 | 4       | 3.2/3.6           |    4      | 1,932  (4,748) | 32GB  | ssd:120GB(/) | hdd:400GB | none             |
| *fiz0*             | W5590     | 4       | 3.33              |    4      | 1,564  (3,342) | 48GB  | ssd:120GB(/) | hdd:400GB | Titan V 24GB     |
|                    |           |         |                   |           |                |       |              |           |                  |
| *vax*              | i3-3220   | 2       | 3.3               |    4      | 1,728  (2,262) | 24GB  | ssd:60GB(/)  | raid10:14TB | fileserver, do not use |
|                    |           |         |                   |           |                |       |              |           |                  |
| *hydra2*           | E5-2699   | 22      | 2.2/3.6           |    44     | 1,896 (20,308) |1024GB | none         | none      | GTX 1080 Ti 11GB  (qty: 4) |

**Note:** hydra2 is a shared compute server with the entire SoC lab. It presently does not mount filesystems exported by vax (and it is difficult to have this done). The GPUs typically go unused, but check for other CPU loads.

**Note:** commands to inspect current workload: ``htop`` (cpu load), ``iotop`` and ``bwm-ng -i disk`` (disk load), ``nload`` and ``bwm-ng`` (network load)



| * Filesystem*      | *Preferred Name (Alias)*    | *Purpose*    |
|--------------------|-----------------------------|--------------|
| /nfs/p3109         | -          | shared 14TB RAID array   |
| /nfs/p3109/home    | /home      | shared home directories  |
| /nfs/p3109/boris   | /nfs/boris | shared project files (eg, training data)  |
| /nfs/p3109/opt     | /nfs/opt   | shared executable programs (eg, Vivado), no host-specific configuration files  |
| /opt               | /opt       | local executable programs (eg, Vivado), no host-specific configuration files  |
| /local/disk1       | -          | fastest local disk (usually the SSD holding root, usually limited size)  |
| /local/disk2       | -          | second local disk (SSD or HDD) |

