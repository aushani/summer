// adpated from devkit_object.zip from KITTI

#include <stdio.h>
#include <stdarg.h>
#include <string.h>

class Mail {
 public:
  Mail() {
    mail = fopen("eval_out.txt", "w");
    fprintf(mail, "From: noreply@cvlibs.net\n");
    fprintf(mail, "Subject: KITTI Evaluation Benchmark\n");
    fprintf(mail, "\n\n");
  }

  ~Mail() {
    if (mail) {
      fclose(mail);
    }
  }

  void msg(const char *format, ...) {
    va_list args;
    va_start(args, format);
    if (mail) {
      vfprintf(mail, format, args);
      fprintf(mail, "\n");
    }
    vprintf(format, args);
    printf("\n");
    va_end(args);
  }

 private:
  FILE *mail = NULL;
};
