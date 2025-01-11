/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/lite/delegates/systolic_sim/systolic_sim_delegate.h"

#include <stdio.h>

#include <memory>
#include <utility>

#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/delegates/utils/simple_delegate.h"

namespace tflite {
namespace dummy_test {

// Dummy delegate kernel.
class SystolicSimDelegateKernel : public SimpleDelegateKernelInterface {
 public:
  explicit SystolicSimDelegateKernel(const SystolicSimDelegateOptions& options)
      : options_(options) {}

  TfLiteStatus Init(TfLiteContext* context,
                    const TfLiteDelegateParams* params) override {
    return !options_.error_during_init ? kTfLiteOk : kTfLiteError;
  }

  TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) override {
    return !options_.error_during_prepare ? kTfLiteOk : kTfLiteError;
  }

  TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) override {
    return !options_.error_during_invoke ? kTfLiteOk : kTfLiteError;
  }

 private:
  const SystolicSimDelegateOptions options_;
};

// DummyDelegate implements the interface of SimpleDelegateInterface.
// This holds the Delegate capabilities.
class SystolicSimDelegate : public SimpleDelegateInterface {
 public:
  explicit SystolicSimDelegate(const SystolicSimDelegateOptions& options)
      : options_(options) {}
  bool IsNodeSupportedByDelegate(const TfLiteRegistration* registration,
                                 const TfLiteNode* node,
                                 TfLiteContext* context) const override {
    if (options_.print_log_to_stderr)
      fprintf(stderr, "We got there!!!!!!!!!!!!!!!\n");

    bool res = false;

    switch (registration->builtin_code) {
      case kTfLiteBuiltinConv2d:
        res = true;
    }

    if (options_.print_log_to_stderr)
      fprintf(stderr, "Operation %s support %d", registration->custom_name,
              res);

    return res;
    // return options_.allowed_builtin_code == registration->builtin_code;
  }

  TfLiteStatus Initialize(TfLiteContext* context) override { return kTfLiteOk; }

  const char* Name() const override {
    static constexpr char kName[] = "DummyDelegate";
    return kName;
  }

  std::unique_ptr<SimpleDelegateKernelInterface> CreateDelegateKernelInterface()
      override {
    return std::make_unique<SystolicSimDelegateKernel>(options_);
  }

  SimpleDelegateInterface::Options DelegateOptions() const override {
    // Use default options.
    return SimpleDelegateInterface::Options();
  }

 private:
  const SystolicSimDelegateOptions options_;
};

}  // namespace dummy_test
}  // namespace tflite

SystolicSimDelegateOptions TfLiteDummyDelegateOptionsDefault() {
  SystolicSimDelegateOptions options = {0};
  // Just assign an invalid builtin code so that this dummy test delegate will
  // not support any node by default.
  options.allowed_builtin_code = -1;
  return options;
}

// Creates a new delegate instance that need to be destroyed with
// `TfLiteDummyDelegateDelete` when delegate is no longer used by TFLite.
// When `options` is set to `nullptr`, the above default values are used:
TfLiteDelegate* TfLiteDummyDelegateCreate(
    const SystolicSimDelegateOptions* options) {
  std::unique_ptr<tflite::dummy_test::SystolicSimDelegate> dummy(
      new tflite::dummy_test::SystolicSimDelegate(
          options ? *options : TfLiteDummyDelegateOptionsDefault()));
  return tflite::TfLiteDelegateFactory::CreateSimpleDelegate(std::move(dummy));
}

// Destroys a delegate created with `TfLiteDummyDelegateCreate` call.
void TfLiteDummyDelegateDelete(TfLiteDelegate* delegate) {
  tflite::TfLiteDelegateFactory::DeleteSimpleDelegate(delegate);
}
