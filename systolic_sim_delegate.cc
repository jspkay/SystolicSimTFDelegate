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
#include <vector>

#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/delegates/utils/simple_delegate.h"
// #include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/tools/logging.h"

namespace tflite {

void printDim(TfLiteIntArray* dims) {
  char res[200] = "(";
  char* ptr = res;
  ptr++;

  if (dims != nullptr) {
    for (int i = 0; i < dims->size; i++) {
      char len[10] = "";
      sprintf(len, "%8d ", dims->data[i]);
      for (int j = 0; j < 9; j++) {
        *ptr = len[j];
        ptr++;
      }
    }
    ptr++;
    *ptr = 0;
    printf("%s )", res);
  } else {
    printf("(NULL)");
  }
}

void printTensor(TfLiteTensor* tensor) {
  auto dims = tensor->dims;
  int n = dims->size;

  std::vector<int> itx;

  itx.resize(n);
  for (int i = 0; i < n; i++) {
    itx.push_back(0);
  }

  int j = 0;
  int k = 0;
  while (true) {
    printf("%f ", tensor->data.f[j]);
    j++;

    if (itx[k] < dims->data[k]) {
      itx[k]++;
    } else {
      k++;
      printf("\n");
    }
    if (k >= n) break;
  }
}

// Dummy delegate kernel.
class SystolicSimDelegateKernel : public SimpleDelegateKernelInterface {
 public:
  explicit SystolicSimDelegateKernel(const SystolicSimDelegateOptions& options)
      : options_(options) {}

  TfLiteStatus Init(TfLiteContext* context,
                    const TfLiteDelegateParams* params) override {
    {
      /*
      printf("Here are all the %d tensors!\n", context->tensors_size);
      for (int i = 0; i < context->tensors_size; i++) {
        printf("%d", i);
        printDim(context->tensors[i].dims);
        printf("%s %d\n", context->tensors[i].name,
               context->tensors[i].allocation_type);
      } */
    }

    TfLiteIntArray* execution_plan;
    context->GetExecutionPlan(context, &execution_plan);
    printf("Execution plan is: ");
    for (int i = 0; i < execution_plan->size; i++) {
      printf("%d ", execution_plan->data[i]);
    }
    printf("\n");

    for (int i = 0; i < execution_plan->size; i++) {
      int j = execution_plan->data[i];
      TfLiteNode** node;
      TfLiteRegistration** reg;
      context->GetNodeAndRegistration(context, j, node, reg);
      printf("Node %d has name \n", j);
    }

    printf("Nodes to replace: %d\n", params->nodes_to_replace->size);
    for (int i = 0; i < params->nodes_to_replace->size; i++) {
      int j = params->nodes_to_replace->data[i];
      printf("Replace node %d with name %s\n", j, context->tensors[j].name);
    }

    inputs_.resize(params->nodes_to_replace->size);
    outputs_.resize(params->nodes_to_replace->size);
    builtin_code_.resize(params->nodes_to_replace->size);
    // For all the nodes to replace
    for (int i = 0; i < params->nodes_to_replace->size; ++i) {
      // take its (global) index
      const int node_index = params->nodes_to_replace->data[i];
      // So that we can Get this node information.
      TfLiteNode* delegated_node = nullptr;
      TfLiteRegistration* delegated_node_registration = nullptr;
      TF_LITE_ENSURE_EQ(
          context,
          context->GetNodeAndRegistration(context, node_index, &delegated_node,
                                          &delegated_node_registration),
          kTfLiteOk);

      // At this point we have a delegated node in delegated_node and a
      // registration in delegated_node_registration
      if (options_.print_log_to_stdout)
        fprintf(stdout, "Got it! Registering node %x\n", delegated_node);

      inputs_[i].push_back(delegated_node->inputs->data[0]);
      inputs_[i].push_back(delegated_node->inputs->data[1]);
      outputs_[i].push_back(delegated_node->outputs->data[0]);
      builtin_code_[i] = delegated_node_registration->builtin_code;
      nodeOperation =
          (TfLiteBuiltinOperator)delegated_node_registration->builtin_code;
      printf("This instance will execute %d\n", nodeOperation);
    }

    return !options_.error_during_init ? kTfLiteOk : kTfLiteError;
  }

  TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) override {
    printf("Preparing node %x to execute operation %d\n", node, nodeOperation);
    return !options_.error_during_prepare ? kTfLiteOk : kTfLiteError;
  }

  TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) override {
    // TODO Change the object type to TfLiteRegistrationExternal (see
    // https://ai.google.dev/edge/api/tflite/c/struct/tf-lite-registration)

    printf("[%x] Evaluating node %x to execute operation %d\n", this, node,
           nodeOperation);

    printf("Ci sono un totale di %d inputs\n", node->inputs->size);

    if (nodeOperation == kTfLiteBuiltinConv2d) {
      printf("Convolutional node\n");

      for (int j = 0; j < node->inputs->size; j++) {
        int i = node->inputs->data[j];
        auto tensor = context->tensors[i];
        printf("-> iteration %i\n", j);
        printDim(tensor.dims);
        std::printf("[%d] Your tensor ha address %x\n", i, &tensor);
        printf("This tensor has allocation type: %d\n",
               (int)tensor.allocation_type);
        printf("The tensor has name %s\n", tensor.name);
        printf("The tensor has type %d\n\n", tensor.type);
        // printTensor(&tensor);
      }
    }
    if (nodeOperation == kTfLiteBuiltinFullyConnected) {
      printf("######## FC ##########\n");
      for (int j = 0; j < node->inputs->size; j++) {
        int i = node->inputs->data[j];
        auto tensor = context->tensors[i];
        printDim(tensor.dims);
        printf("-> iteration %i\n", j);
        std::printf("[%i] Your tensor ha address %x\n", i, &tensor);
        printf("This tensor has allocation type: %d\n",
               (int)tensor.allocation_type);
        printf("The tensor has name %s\n", tensor.name);
        printf("The tensor has type %d\n\n", tensor.type);
      }
    }

    auto size = context->tensors_size;

    return !options_.error_during_invoke ? kTfLiteOk : kTfLiteError;
  }

 private:
  const SystolicSimDelegateOptions options_;
  std::vector<std::vector<int>> inputs_, outputs_;
  std::vector<int> builtin_code_;

  TfLiteBuiltinOperator nodeOperation;
};

// SystolicSimDelegate implements the interface of SimpleDelegateInterface.
// This holds the Delegate capabilities.
class SystolicSimDelegate : public SimpleDelegateInterface {
 public:
  explicit SystolicSimDelegate(const SystolicSimDelegateOptions& options)
      : options_(options) {}
  bool IsNodeSupportedByDelegate(const TfLiteRegistration* registration,
                                 const TfLiteNode* node,
                                 TfLiteContext* context) const override {
    bool res = false;

    switch (registration->builtin_code) {
      case kTfLiteBuiltinConv2d:
        res = true;
        break;
      case kTfLiteBuiltinFullyConnected:
        res = true;
        break;
    }

    if (options_.print_log_to_stdout)
      fprintf(stdout, "Operation %i support %d\n", registration->builtin_code,
              res);

    return res;
    // return options_.allowed_builtin_code == registration->builtin_code;
  }

  TfLiteStatus Initialize(TfLiteContext* context) override {
    TfLiteIntArray** execution_plan;
    context->GetExecutionPlan(context, execution_plan);
    return kTfLiteOk;
  }

  const char* Name() const override {
    static constexpr char kName[] = "SystolicSimDelegate";
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
  std::unique_ptr<tflite::SystolicSimDelegate> dummy(
      new tflite::SystolicSimDelegate(
          options ? *options : TfLiteDummyDelegateOptionsDefault()));
  return tflite::TfLiteDelegateFactory::CreateSimpleDelegate(std::move(dummy));
}

// Destroys a delegate created with `TfLiteDummyDelegateCreate` call.
void TfLiteDummyDelegateDelete(TfLiteDelegate* delegate) {
  tflite::TfLiteDelegateFactory::DeleteSimpleDelegate(delegate);
}
