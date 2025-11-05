#include "matgen/util/argparse.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "matgen/util/log.h"

#define MAX_ARGS 64

// =============================================================================
// Internal Structures
// =============================================================================

struct matgen_argparser {
  const char* program_name;
  const char* description;
  matgen_arg_t args[MAX_ARGS];
  size_t num_args;
};

// =============================================================================
// Parser Creation and Destruction
// =============================================================================

matgen_argparser_t* matgen_argparser_create(const char* program_name,
                                            const char* description) {
  if (!program_name) {
    return NULL;
  }

  matgen_argparser_t* parser =
      (matgen_argparser_t*)malloc(sizeof(matgen_argparser_t));
  if (!parser) {
    return NULL;
  }

  parser->program_name = program_name;
  parser->description = description;
  parser->num_args = 0;

  // Add default help flag
  bool* help_flag = (bool*)calloc(1, sizeof(bool));
  matgen_argparser_add_flag(parser, "h", "help", help_flag,
                            "Show this help message");

  return parser;
}

void matgen_argparser_destroy(matgen_argparser_t* parser) {
  if (!parser) {
    return;
  }
  free(parser);
}

// =============================================================================
// Adding Arguments
// =============================================================================

int matgen_argparser_add(matgen_argparser_t* parser, const matgen_arg_t* arg) {
  if (!parser || !arg) {
    return -1;
  }

  if (parser->num_args >= MAX_ARGS) {
    MATGEN_LOG_ERROR("Maximum number of arguments (%d) reached", MAX_ARGS);
    return -1;
  }

  parser->args[parser->num_args++] = *arg;
  return 0;
}

int matgen_argparser_add_flag(matgen_argparser_t* parser, const char* short_opt,
                              const char* long_opt, bool* dest,
                              const char* help) {
  matgen_arg_t arg = {.short_opt = short_opt,
                      .long_opt = long_opt,
                      .type = MATGEN_ARG_BOOL,
                      .dest = dest,
                      .help = help,
                      .default_value = NULL,
                      .required = false};

  // Initialize to false
  if (dest) {
    *dest = false;
  }

  return matgen_argparser_add(parser, &arg);
}

int matgen_argparser_add_int(matgen_argparser_t* parser, const char* short_opt,
                             const char* long_opt, int* dest, int default_val,
                             const char* help) {
  static char default_buf[32];
  snprintf(default_buf, sizeof(default_buf), "%d", default_val);

  matgen_arg_t arg = {.short_opt = short_opt,
                      .long_opt = long_opt,
                      .type = MATGEN_ARG_INT,
                      .dest = dest,
                      .help = help,
                      .default_value = default_buf,
                      .required = false};

  // Set default value
  if (dest) {
    *dest = default_val;
  }

  return matgen_argparser_add(parser, &arg);
}

int matgen_argparser_add_string(matgen_argparser_t* parser,
                                const char* short_opt, const char* long_opt,
                                const char** dest, const char* default_val,
                                const char* help) {
  matgen_arg_t arg = {.short_opt = short_opt,
                      .long_opt = long_opt,
                      .type = MATGEN_ARG_STRING,
                      .dest = (void*)dest,
                      .help = help,
                      .default_value = default_val,
                      .required = false};

  // Set default value
  if (dest) {
    *dest = default_val;
  }

  return matgen_argparser_add(parser, &arg);
}

// =============================================================================
// Parsing Implementation
// =============================================================================

static matgen_arg_t* find_arg_by_short(matgen_argparser_t* parser,
                                       const char* opt) {
  for (size_t i = 0; i < parser->num_args; i++) {
    if (parser->args[i].short_opt &&
        strcmp(parser->args[i].short_opt, opt) == 0) {
      return &parser->args[i];
    }
  }
  return NULL;
}

static matgen_arg_t* find_arg_by_long(matgen_argparser_t* parser,
                                      const char* opt) {
  for (size_t i = 0; i < parser->num_args; i++) {
    if (parser->args[i].long_opt &&
        strcmp(parser->args[i].long_opt, opt) == 0) {
      return &parser->args[i];
    }
  }
  return NULL;
}

static int parse_value(matgen_arg_t* arg, const char* value) {
  if (!arg || !value) {
    return -1;
  }

  switch (arg->type) {
    case MATGEN_ARG_BOOL:
      *(bool*)arg->dest = true;
      break;

    case MATGEN_ARG_INT: {
      char* endptr;
      long val = strtol(value, &endptr, 10);
      if (*endptr != '\0') {
        MATGEN_LOG_ERROR("Invalid integer value: %s", value);
        return -1;
      }
      *(int*)arg->dest = (int)val;
      break;
    }

    case MATGEN_ARG_DOUBLE: {
      char* endptr;
      double val = strtod(value, &endptr);
      if (*endptr != '\0') {
        MATGEN_LOG_ERROR("Invalid double value: %s", value);
        return -1;
      }
      *(double*)arg->dest = val;
      break;
    }

    case MATGEN_ARG_STRING:
      *(const char**)arg->dest = value;
      break;

    default:
      return -1;
  }

  return 0;
}

// NOLINTNEXTLINE
int matgen_argparser_parse(matgen_argparser_t* parser, int argc, char** argv) {
  if (!parser || argc < 1) {
    return -1;
  }

  for (int i = 1; i < argc; i++) {
    const char* arg_str = argv[i];

    // Long option: --option or --option=value
    if (strncmp(arg_str, "--", 2) == 0) {
      const char* opt_name = arg_str + 2;
      const char* eq = strchr(opt_name, '=');

      char opt_buf[64];
      if (eq) {
        // --option=value format
        size_t len = eq - opt_name;
        if (len >= sizeof(opt_buf)) {
          MATGEN_LOG_ERROR("Option name too long: %s", arg_str);
          return -1;
        }
        strncpy(opt_buf, opt_name, len);
        opt_buf[len] = '\0';
        opt_name = opt_buf;
      }

      matgen_arg_t* arg = find_arg_by_long(parser, opt_name);
      if (!arg) {
        MATGEN_LOG_ERROR("Unknown option: --%s", opt_name);
        return -1;
      }

      // Handle value
      if (arg->type == MATGEN_ARG_BOOL) {
        *(bool*)arg->dest = true;
      } else {
        const char* value;
        if (eq) {
          value = eq + 1;
        } else {
          if (i + 1 >= argc) {
            MATGEN_LOG_ERROR("Option --%s requires a value", opt_name);
            return -1;
          }
          value = argv[++i];
        }

        if (parse_value(arg, value) != 0) {
          return -1;
        }
      }
    }
    // Short option: -o or -o value
    else if (arg_str[0] == '-' && arg_str[1] != '\0') {
      const char* opt_name = arg_str + 1;

      matgen_arg_t* arg = find_arg_by_short(parser, opt_name);
      if (!arg) {
        MATGEN_LOG_ERROR("Unknown option: -%s", opt_name);
        return -1;
      }

      // Handle value
      if (arg->type == MATGEN_ARG_BOOL) {
        *(bool*)arg->dest = true;
      } else {
        if (i + 1 >= argc) {
          MATGEN_LOG_ERROR("Option -%s requires a value", opt_name);
          return -1;
        }
        const char* value = argv[++i];

        if (parse_value(arg, value) != 0) {
          return -1;
        }
      }
    }
    // Positional argument
    else {
      MATGEN_LOG_WARN("Ignoring positional argument: %s", arg_str);
    }
  }

  // Check if help was requested
  for (size_t i = 0; i < parser->num_args; i++) {
    if (parser->args[i].type == MATGEN_ARG_BOOL && parser->args[i].long_opt &&
        strcmp(parser->args[i].long_opt, "help") == 0) {
      if (*(bool*)parser->args[i].dest) {
        matgen_argparser_print_help(parser, stdout);
        exit(0);
      }
    }
  }

  return 0;
}

// =============================================================================
// Help Generation
// =============================================================================

void matgen_argparser_print_usage(const matgen_argparser_t* parser,
                                  FILE* stream) {
  if (!parser || !stream) {
    return;
  }

  fprintf(stream, "Usage: %s [OPTIONS]\n", parser->program_name);
}

void matgen_argparser_print_help(const matgen_argparser_t* parser,
                                 FILE* stream) {
  if (!parser || !stream) {
    return;
  }

  matgen_argparser_print_usage(parser, stream);

  if (parser->description) {
    fprintf(stream, "\n%s\n", parser->description);
  }

  fprintf(stream, "\nOptions:\n");

  for (size_t i = 0; i < parser->num_args; i++) {
    const matgen_arg_t* arg = &parser->args[i];

    fprintf(stream, "  ");

    if (arg->short_opt) {
      fprintf(stream, "-%s", arg->short_opt);
      if (arg->long_opt) {
        fprintf(stream, ", ");
      }
    } else {
      fprintf(stream, "    ");
    }

    if (arg->long_opt) {
      fprintf(stream, "--%s", arg->long_opt);
    }

    // Add type hint
    switch (arg->type) {
      case MATGEN_ARG_INT:
        fprintf(stream, " <int>");
        break;
      case MATGEN_ARG_DOUBLE:
        fprintf(stream, " <double>");
        break;
      case MATGEN_ARG_STRING:
        fprintf(stream, " <string>");
        break;
      default:
        break;
    }

    fprintf(stream, "\n      %s", arg->help ? arg->help : "");

    if (arg->default_value) {
      fprintf(stream, " (default: %s)", arg->default_value);
    }

    fprintf(stream, "\n");
  }
}
